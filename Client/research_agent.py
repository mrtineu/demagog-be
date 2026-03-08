# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai",
#     "python-dotenv",
#     "tavily-python",
# ]
# ///

"""
Web research agent for Demagog.sk fact-checking.

Activated when the DB verification pipeline returns "Nedostatok dat" (no match).
Searches trusted websites via Tavily, extracts content, and uses Gemini 2.5 Pro
to produce a verdict with source citations.
"""

import json
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

from shared.verdicts import VERDICT_LABEL
from shared.prompts import RESEARCH_SYSTEM_PROMPT, QUERY_GENERATION_PROMPT

load_dotenv()

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "backend" / "config" / "research_config.json"

MAX_CONTENT_LENGTH = 3000  # chars per source to prevent context overflow


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load research configuration from JSON file."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_all_trusted_domains(config: dict) -> list[str]:
    """Flatten all trusted domain categories into a single list."""
    all_domains: list[str] = []
    for category_domains in config["trusted_domains"].values():
        all_domains.extend(category_domains)
    return list(set(all_domains))  # deduplicate


# ---------------------------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------------------------

def get_openrouter_client() -> OpenAI:
    """Create OpenRouter client using .env key."""
    api_key = os.getenv("OPENROUTER_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_KEY not found in .env file.")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------


def generate_search_queries(
    statement: str, client: OpenAI, config: dict
) -> list[str]:
    """Use LLM to transform a Slovak political claim into optimized search queries.

    Returns 2-3 keyword-style queries in Slovak and English.
    Falls back to [statement] on any failure.
    """
    llm_settings = config["llm_settings"]

    try:
        response = client.chat.completions.create(
            model=llm_settings["model"],
            temperature=0.0,
            max_tokens=256,
            messages=[
                {"role": "system", "content": QUERY_GENERATION_PROMPT},
                {"role": "user", "content": statement},
            ],
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw = "\n".join(lines).strip()

        queries = json.loads(raw)

        if (
            isinstance(queries, list)
            and len(queries) >= 1
            and all(isinstance(q, str) and q.strip() for q in queries)
        ):
            return [q.strip() for q in queries[:5]]

    except Exception as e:
        logger.warning("Query generation failed: %s", e)

    return [statement]


# ---------------------------------------------------------------------------
# Web search (Tavily)
# ---------------------------------------------------------------------------

def filter_excluded(results: list[dict], excluded: list[str]) -> list[dict]:
    """Remove any results whose domain matches the exclusion list."""
    filtered = []
    for r in results:
        domain = urlparse(r.get("url", "")).netloc.lower()
        is_excluded = any(
            domain == ex or domain.endswith("." + ex)
            for ex in excluded
        )
        if not is_excluded:
            filtered.append(r)
        else:
            logger.info("Filtered excluded domain: %s", domain)
    return filtered


def search_web(
    statement: str, config: dict, queries: list[str] | None = None
) -> list[dict]:
    """
    Two-phase Tavily search with multi-query support:
    1. Restricted to trusted domains only.
    2. Fallback: broad search with excluded domains filtered out.
    When *queries* is provided, each query is searched and results are merged
    (deduplicated by URL).  Falls back to [statement] when queries is None.
    Returns list of result dicts with url, title, content, score.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not found in .env")

    tavily = TavilyClient(api_key=api_key)
    settings = config["search_settings"]
    trusted = get_all_trusted_domains(config)
    excluded = config["excluded_domains"]
    topic = settings.get("topic", "general")

    search_queries = queries if queries else [statement]
    all_results: list[dict] = []
    seen_urls: set[str] = set()

    for query in search_queries:
        # Phase 1: Trusted-only search
        try:
            phase1 = tavily.search(
                query=query,
                search_depth=settings["search_depth"],
                topic=topic,
                max_results=settings["max_results"],
                include_domains=trusted,
            )
            for r in phase1.get("results", []):
                if r["url"] not in seen_urls:
                    all_results.append(r)
                    seen_urls.add(r["url"])
        except Exception as e:
            logger.warning("Tavily Phase 1 failed for query '%s': %s", query, e)

        # Phase 2: Broad search if still insufficient
        if len(all_results) < settings["min_sources_for_verdict"]:
            try:
                phase2 = tavily.search(
                    query=query,
                    search_depth=settings["search_depth"],
                    topic=topic,
                    max_results=settings["max_results"],
                    exclude_domains=excluded,
                )
                phase2_results = phase2.get("results", [])
                phase2_results = filter_excluded(phase2_results, excluded)
                for r in phase2_results:
                    if r["url"] not in seen_urls:
                        all_results.append(r)
                        seen_urls.add(r["url"])
            except Exception as e:
                logger.warning("Tavily Phase 2 failed for query '%s': %s", query, e)

    return all_results


# ---------------------------------------------------------------------------
# Content extraction (Tavily Extract)
# ---------------------------------------------------------------------------

def extract_content(
    results: list[dict], statement: str, config: dict
) -> list[dict]:
    """
    Extract full content from top search result URLs using Tavily Extract.
    Returns enriched result dicts with 'raw_content' field added.
    """
    if not results:
        return []

    api_key = os.getenv("TAVILY_API_KEY")
    tavily = TavilyClient(api_key=api_key)
    settings = config["search_settings"]

    max_extract = settings.get("max_extract_urls", 5)
    urls_to_extract = [r["url"] for r in results[:max_extract]]

    try:
        extract_response = tavily.extract(urls=urls_to_extract)
    except Exception as e:
        logger.warning("Tavily extract failed, using snippets: %s", e)
        return results

    extracted_map = {
        item["url"]: item.get("raw_content", "")
        for item in extract_response.get("results", [])
    }

    for r in results:
        if r["url"] in extracted_map:
            r["raw_content"] = extracted_map[r["url"]]
        else:
            r["raw_content"] = r.get("content", "")  # fallback to snippet

    return results


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def build_research_user_message(
    statement: str, enriched_results: list[dict]
) -> str:
    """Format the web research results into an LLM user message."""
    parts = [
        f'Politický výrok na overenie: "{statement}"\n\n'
        f"Nájdených webových zdrojov: {len(enriched_results)}\n\n"
        f"=== WEBOVÉ ZDROJE ===\n"
    ]

    for i, r in enumerate(enriched_results, 1):
        content = r.get("raw_content", r.get("content", ""))
        if len(content) > MAX_CONTENT_LENGTH:
            content = content[:MAX_CONTENT_LENGTH] + "... [skrátené]"

        parts.append(
            f"\n--- Zdroj {i} ---\n"
            f"URL: {r.get('url', 'N/A')}\n"
            f"Názov: {r.get('title', 'N/A')}\n"
            f"Úryvok z vyhľadávania: {r.get('content', 'N/A')}\n"
            f"Plný obsah:\n{content}\n"
        )

    return "\n".join(parts)


def call_research_llm(
    client: OpenAI,
    statement: str,
    enriched_results: list[dict],
    config: dict,
) -> dict:
    """Send web research results to LLM for fact-check analysis."""
    user_message = build_research_user_message(statement, enriched_results)
    llm_settings = config["llm_settings"]

    response = client.chat.completions.create(
        model=llm_settings["model"],
        temperature=llm_settings["temperature"],
        max_tokens=llm_settings.get("max_tokens", 4096),
        messages=[
            {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    raw_content = response.choices[0].message.content.strip()

    # Strip markdown code fences if present (same pattern as verify.py)
    if raw_content.startswith("```"):
        lines = raw_content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw_content = "\n".join(lines).strip()

    return json.loads(raw_content)


# ---------------------------------------------------------------------------
# Conservative safeguards (programmatic enforcement)
# ---------------------------------------------------------------------------

def enforce_conservative_safeguards(llm_response: dict) -> dict:
    """
    Programmatic enforcement of conservative fact-checking rules.
    Overrides LLM verdict if safeguard conditions are not met.
    """
    verdikt = llm_response.get("verdikt", "Neoveriteľné")
    sources = llm_response.get("pouzite_zdroje", [])
    num_sources = llm_response.get("pocet_podpornych_zdrojov", 0)
    confidence = llm_response.get("istota", "nízka")
    contradictions = llm_response.get("protirecie")

    override_reason = None
    definitive_verdicts = {"Pravda", "Nepravda", "Zavádzajúce"}

    # Rule 1: Must have 2+ independent sources for a definitive verdict
    if verdikt in definitive_verdicts and num_sources < 2:
        override_reason = (
            f"LLM navrhol verdikt '{verdikt}' s iba {num_sources} zdrojom/zdrojmi. "
            f"Pravidlo dvoch nezávislých zdrojov nebolo splnené."
        )

    # Rule 2: If sources contradict each other, force Neoveriteľné
    if verdikt in definitive_verdicts and contradictions:
        override_reason = (
            f"LLM navrhol verdikt '{verdikt}' napriek zisteným rozporom "
            f"medzi zdrojmi: {contradictions}"
        )

    # Rule 3: Low confidence with definitive verdict
    if verdikt in definitive_verdicts and confidence == "nízka":
        override_reason = (
            f"LLM navrhol verdikt '{verdikt}' s nízkou istotou. "
            f"Pri nízkej istote sa vyžaduje verdikt 'Neoveriteľné'."
        )

    # Rule 4: Verify source records match the claimed count
    if verdikt in definitive_verdicts and len(sources) < 2:
        override_reason = (
            f"LLM uviedol {num_sources} podporných zdrojov, "
            f"ale poskytol iba {len(sources)} záznamov."
        )

    if override_reason:
        logger.warning("Safeguard override: %s", override_reason)
        llm_response["povodny_verdikt_llm"] = verdikt
        llm_response["verdikt"] = "Neoveriteľné"
        llm_response["safeguard_override"] = True
        llm_response["dovod_override"] = override_reason
        llm_response["istota"] = "nízka"
    else:
        llm_response["safeguard_override"] = False

    return llm_response


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------

def build_research_result(
    statement: str,
    llm_response: dict,
    search_results_count: int,
) -> dict:
    """Build the final structured result from web research."""
    verdikt = llm_response.get("verdikt", "Neoveriteľné")

    return {
        # Fields matching verify.py output format
        "vstupny_vyrok": statement,
        "status": "webovy_vyskum",
        "verdikt": verdikt,
        "verdikt_label": VERDICT_LABEL.get(verdikt, verdikt),
        "odovodnenie_llm": llm_response.get("odovodnenie_llm", ""),
        "zdrojovy_vyrok": None,
        "zdroj": None,
        # Web research specific fields
        "typ_overenia": "webovy_vyskum",
        "istota": llm_response.get("istota", "nízka"),
        "webove_zdroje": llm_response.get("pouzite_zdroje", []),
        "pocet_najdenych_zdrojov": search_results_count,
        "pocet_podpornych_zdrojov": llm_response.get(
            "pocet_podpornych_zdrojov", 0
        ),
        "protirecie": llm_response.get("protirecie"),
        "safeguard_override": llm_response.get("safeguard_override", False),
        # Compatibility fields (mimic verify.py shape)
        "pouzity_prah": None,
        "pocet_nad_prahom": 0,
        "pocet_celkom": 0,
    }


def _make_no_data_response(reason: str) -> dict:
    """Helper: construct an LLM-response-shaped dict for error/no-data cases."""
    return {
        "verdikt": "Neoveriteľné",
        "istota": "nízka",
        "odovodnenie_llm": reason,
        "pouzite_zdroje": [],
        "pocet_podpornych_zdrojov": 0,
        "protirecie": None,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def research_statement(
    statement: str, llm_client: OpenAI | None = None
) -> dict:
    """
    Full web research pipeline for a political statement.
    Called when verify.py's DB search returns no match.

    Returns a result dict compatible with verify.py's output format,
    with additional web research fields.
    """
    config = load_config()

    if llm_client is None:
        llm_client = get_openrouter_client()

    # Step 0: Generate optimized search queries
    try:
        queries = generate_search_queries(statement, llm_client, config)
        logger.info("Generated search queries: %s", queries)
    except Exception as e:
        logger.warning("Query generation failed, using raw statement: %s", e)
        queries = None

    # Step 1: Web search (two-phase, multi-query)
    try:
        search_results = search_web(statement, config, queries=queries)
    except Exception as e:
        logger.error("Web search failed completely: %s", e)
        return build_research_result(
            statement,
            _make_no_data_response(
                "Webové vyhľadávanie zlyhalo. "
                "Nie je možné overiť výrok bez prístupu k zdrojom."
            ),
            search_results_count=0,
        )

    # Step 2: Check if any results were found
    if not search_results:
        return build_research_result(
            statement,
            _make_no_data_response(
                "Webové vyhľadávanie nenašlo žiadne relevantné zdroje "
                "na overenie tohto výroku."
            ),
            search_results_count=0,
        )

    # Step 3: Extract full content from top results
    try:
        enriched = extract_content(search_results, statement, config)
    except Exception as e:
        logger.warning("Content extraction failed, using snippets: %s", e)
        enriched = search_results

    # Step 4: LLM analysis
    try:
        llm_response = call_research_llm(
            llm_client, statement, enriched, config
        )
    except json.JSONDecodeError as e:
        logger.error("LLM returned invalid JSON: %s", e)
        return build_research_result(
            statement,
            _make_no_data_response(
                "Analýza webových zdrojov zlyhala (neplatná odpoveď LLM). "
                "Nie je možné poskytnúť verdikt."
            ),
            search_results_count=len(search_results),
        )
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return build_research_result(
            statement,
            _make_no_data_response(f"Volanie LLM zlyhalo: {e}"),
            search_results_count=len(search_results),
        )

    # Step 5: Enforce conservative safeguards
    llm_response = enforce_conservative_safeguards(llm_response)

    # Step 6: Build final result
    return build_research_result(
        statement, llm_response, search_results_count=len(search_results)
    )
