# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "requests",
#     "qdrant-client",
#     "openai",
#     "python-dotenv",
#     "tavily-python",
# ]
# ///

import argparse
import io
import json
import os
import sys

# Ensure stdin/stdout handle Slovak characters correctly regardless of locale
if hasattr(sys.stdin, "buffer"):
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from openai import OpenAI
import requests
from qdrant_client import QdrantClient

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent))
from shared.verdicts import VERDICT_LABEL
from shared.prompts import VERIFY_SYSTEM_PROMPT as SYSTEM_PROMPT

from research_agent import research_statement

load_dotenv()

# Infrastructure
EC2_IP = "13.48.59.38"
QDRANT_PORT = 6333
INFINITY_PORT = 7997
COLLECTION = "test_1"
MODEL = "BAAI/bge-m3"

# Verification defaults
TOP_K = 5
DEFAULT_THRESHOLD = 0.6
LLM_MODEL = "google/gemini-3-flash-preview"
LLM_TEMPERATURE = 0.1

# Incomplete-statement detection
MIN_WORD_COUNT = 4
MIN_LENGTH_RATIO = 0.5


# --- Infrastructure ---

def embed(text: str) -> list[float]:
    """Get embedding vector via the Infinity API."""
    resp = requests.post(
        f"http://{EC2_IP}:{INFINITY_PORT}/embeddings",
        json={"model": MODEL, "input": text},
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def get_openrouter_client() -> OpenAI:
    """Create OpenRouter client using .env key."""
    api_key = os.getenv("OPENROUTER_KEY")
    if not api_key:
        print("Error: OPENROUTER_KEY not found in .env file.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


# --- Search & Filter ---

def search_similar(qdrant: QdrantClient, query_text: str, top_k: int = TOP_K) -> list[dict]:
    """Embed query and search Qdrant. Returns list of dicts with score + payload fields."""
    vector = embed(query_text)
    results = qdrant.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=top_k,
        with_payload=True,
    ).points

    return [
        {
            "score": hit.score,
            "vyrok": (hit.payload or {}).get("Výrok", ""),
            "vyhodnotenie": (hit.payload or {}).get("Vyhodnotenie", ""),
            "odovodnenie": (hit.payload or {}).get("Odôvodnenie", ""),
            "oblast": (hit.payload or {}).get("Oblast", ""),
            "datum": (hit.payload or {}).get("Dátum", ""),
            "meno": (hit.payload or {}).get("Meno", ""),
            "politicka_strana": (hit.payload or {}).get("Politická strana", ""),
        }
        for hit in results
    ]


def filter_by_threshold(results: list[dict], threshold: float) -> list[dict]:
    """Keep only results with similarity score >= threshold."""
    return [r for r in results if r["score"] >= threshold]


def is_incomplete_statement(statement: str, db_results: list[dict]) -> tuple[bool, str]:
    """Detect obviously incomplete or fragmentary input statements.

    Returns (True, reason) if the statement appears incomplete,
    (False, "") otherwise.
    """
    words = statement.split()

    # Check 1: Minimum word count
    if len(words) < MIN_WORD_COUNT:
        return True, (
            f"Vstupný výrok obsahuje iba {len(words)} slov(á). "
            "Úplné faktické tvrdenie vyžaduje aspoň podmět, prísudok a predmět."
        )

    # Check 2: Input much shorter than best matching DB statement
    if db_results:
        best_match = db_results[0]["vyrok"]
        if best_match:
            ratio = len(statement) / len(best_match)
            if ratio < MIN_LENGTH_RATIO:
                return True, (
                    f"Vstupný výrok je výrazne kratší než najbližší nájdený záznam "
                    f"({len(statement)} vs {len(best_match)} znakov, pomer {ratio:.0%}). "
                    "Neúplný výrok nie je možné overiť."
                )

    return False, ""


# --- LLM Call ---

def build_user_message(input_statement: str, db_results: list[dict]) -> str:
    """Format the user message with the input statement and all DB results."""
    parts = [f'Vstupný výrok: "{input_statement}"\n\nVýsledky z databázy:\n']
    for i, r in enumerate(db_results, 1):
        parts.append(
            f"--- Výsledok {i} (skóre podobnosti: {r['score']:.4f}) ---\n"
            f"Výrok: {r['vyrok']}\n"
            f"Verdikt: {r['vyhodnotenie']}\n"
            f"Odôvodnenie: {r['odovodnenie']}\n"
            f"Politik: {r['meno']} ({r['politicka_strana']})\n"
            f"Oblasť: {r['oblast']}\n"
            f"Dátum: {r['datum']}\n"
        )
    return "\n".join(parts)


def call_llm(client: OpenAI, input_statement: str, db_results: list[dict]) -> dict:
    """Send filtered results to LLM and parse JSON response."""
    user_message = build_user_message(input_statement, db_results)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    raw_content = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw_content.startswith("```"):
        lines = raw_content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw_content = "\n".join(lines).strip()

    return json.loads(raw_content)


# --- Result Assembly ---

def build_verification_result(
    input_statement: str,
    llm_response: dict,
    db_results: list[dict],
    all_raw_results: list[dict],
    threshold: float,
) -> dict:
    """Assemble the full verification result combining LLM analysis with DB metadata."""
    result = {
        "vstupny_vyrok": input_statement,
        "status": "zhoda" if llm_response.get("zhoda") else "bez_zhody",
        "verdikt": llm_response.get("verdikt", "Nedostatok dát"),
        "verdikt_label": VERDICT_LABEL.get(
            llm_response.get("verdikt", ""), llm_response.get("verdikt", "Nedostatok dát")
        ),
        "odovodnenie_llm": llm_response.get("odovodnenie_llm", ""),
        "zdrojovy_vyrok": llm_response.get("zdrojovy_vyrok"),
        "zdroj": None,
        "pouzity_prah": threshold,
        "pocet_nad_prahom": len(db_results),
        "pocet_celkom": len(all_raw_results),
    }

    # If match found, attach full metadata from the matched DB result
    idx = llm_response.get("index_zhody")
    if llm_response.get("zhoda") and idx is not None and 1 <= idx <= len(db_results):
        matched = db_results[idx - 1]
        result["zdroj"] = {
            "vyrok": matched["vyrok"],
            "vyhodnotenie": matched["vyhodnotenie"],
            "odovodnenie": matched["odovodnenie"],
            "oblast": matched["oblast"],
            "datum": matched["datum"],
            "meno": matched["meno"],
            "politicka_strana": matched["politicka_strana"],
            "skore_podobnosti": matched["score"],
        }

    return result


def build_no_data_result(
    input_statement: str, all_raw_results: list[dict], threshold: float
) -> dict:
    """Build structured result when no DB results pass the threshold."""
    top_score = all_raw_results[0]["score"] if all_raw_results else 0.0
    return {
        "vstupny_vyrok": input_statement,
        "status": "bez_zhody",
        "verdikt": "Nedostatok dát",
        "verdikt_label": "NEDOSTATOK DÁT",
        "odovodnenie_llm": (
            f"Žiadny výsledok z databázy nepresiahol prah podobnosti {threshold}. "
            f"Najvyššie skóre: {top_score:.4f}."
        ),
        "zdrojovy_vyrok": None,
        "zdroj": None,
        "pouzity_prah": threshold,
        "pocet_nad_prahom": 0,
        "pocet_celkom": len(all_raw_results),
    }


# --- Output Formatting ---

def print_human_readable(result: dict) -> None:
    """Pretty-print result for terminal."""
    status = result["status"]
    print(f"\n{'=' * 72}")

    if status == "zhoda":
        verdikt = result["verdikt_label"]
        print(f"  VERDIKT: {verdikt}")
        print(f"{'=' * 72}")
        print(f"\n  Vstupný výrok:   {result['vstupny_vyrok']}")
        print(f"  Zdrojový výrok:  {result['zdrojovy_vyrok']}")
        print(f"\n  Odôvodnenie LLM: {result['odovodnenie_llm']}")

        src = result.get("zdroj")
        if src:
            print(f"\n  --- Zdroj z databázy ---")
            print(f"  Politik:         {src['meno']} ({src['politicka_strana']})")
            print(f"  Pôvodný verdikt: {src['vyhodnotenie']}")
            print(f"  Oblasť:          {src['oblast'] or 'N/A'}")
            print(f"  Dátum:           {src['datum'] or 'N/A'}")
            print(f"  Skóre:           {src['skore_podobnosti']:.4f}")
            odov = src.get("odovodnenie", "")
            if odov:
                print(f"  Odôvodnenie:     {odov[:400]}{'...' if len(odov) > 400 else ''}")
    elif status == "webovy_vyskum":
        verdikt = result.get("verdikt_label", result.get("verdikt", "N/A"))
        print(f"  VERDIKT: {verdikt} (webový výskum)")
        print(f"{'=' * 72}")
        print(f"\n  Vstupný výrok:   {result['vstupny_vyrok']}")
        print(f"  Istota:          {result.get('istota', 'N/A')}")
        print(f"\n  Odôvodnenie LLM: {result['odovodnenie_llm']}")

        sources = result.get("webove_zdroje", [])
        if sources:
            print(f"\n  --- Webové zdroje ({len(sources)}) ---")
            for i, src in enumerate(sources, 1):
                print(f"  [{i}] {src.get('nazov', 'N/A')}")
                print(f"      URL: {src.get('url', 'N/A')}")
                print(f"      Typ: {src.get('typ_zdroja', 'N/A')}")
                citat = src.get("relevantny_citat", "")
                if citat:
                    print(f"      Citát: {citat[:200]}{'...' if len(citat) > 200 else ''}")

        if result.get("safeguard_override"):
            print(f"\n  [!] Bezpečnostná poistka aktivovaná")

        print(f"\n  [Nájdené zdroje: {result.get('pocet_najdenych_zdrojov', 0)} | "
              f"Podporné: {result.get('pocet_podpornych_zdrojov', 0)}]")
        print(f"\n  Toto je automatický návrh na základe webového výskumu.")
        print(f"  Konečné overenie musí vykonať ľudský analytik (min. 3-osobná kontrola).")
    else:
        print(f"  VERDIKT: NEDOSTATOK DÁT")
        print(f"{'=' * 72}")
        print(f"\n  Vstupný výrok:   {result['vstupny_vyrok']}")
        print(f"\n  Dôvod:           {result['odovodnenie_llm']}")

    prah = result.get("pouzity_prah")
    if prah is not None:
        print(f"\n  [Prah: {prah} | "
              f"Nad prahom: {result.get('pocet_nad_prahom', 0)}/{result.get('pocet_celkom', 0)}]")
    print(f"{'=' * 72}\n")


def print_json(result: dict) -> None:
    """Print result as formatted JSON."""
    print(json.dumps(result, ensure_ascii=False, indent=2))


# --- Main Loop & CLI ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overovač politických výrokov - fact-check verification"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output results in JSON format instead of human-readable",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Similarity score threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--no-research",
        action="store_true",
        default=False,
        help="Disable web research fallback when no DB match is found",
    )
    return parser.parse_args()


def verify_statement(
    statement: str,
    qdrant: QdrantClient,
    llm_client: OpenAI,
    threshold: float,
    enable_research: bool = True,
) -> dict:
    """Full verification pipeline for a single statement."""
    # Step 1: Search Qdrant
    all_results = search_similar(qdrant, statement, TOP_K)

    if not all_results:
        result = build_no_data_result(statement, all_results, threshold)
        if enable_research:
            return _try_research_fallback(statement, llm_client, result)
        return result

    # Step 2: Filter by threshold
    above_threshold = filter_by_threshold(all_results, threshold)

    if not above_threshold:
        result = build_no_data_result(statement, all_results, threshold)
        if enable_research:
            return _try_research_fallback(statement, llm_client, result)
        return result

    # Step 2.5: Reject incomplete / fragmentary input (NO research for fragments)
    incomplete, reason = is_incomplete_statement(
        statement, above_threshold or all_results
    )
    if incomplete:
        return {
            "vstupny_vyrok": statement,
            "status": "bez_zhody",
            "verdikt": "Nedostatok dát",
            "verdikt_label": "NEDOSTATOK DÁT",
            "odovodnenie_llm": reason,
            "zdrojovy_vyrok": None,
            "zdroj": None,
            "pouzity_prah": threshold,
            "pocet_nad_prahom": len(above_threshold),
            "pocet_celkom": len(all_results),
        }

    # Step 3: LLM evaluation
    llm_response = call_llm(llm_client, statement, above_threshold)

    # Step 4: Assemble result
    result = build_verification_result(
        statement, llm_response, above_threshold, all_results, threshold
    )

    # Step 5: If LLM found no DB match, try web research
    if enable_research and result["verdikt"] == "Nedostatok dát":
        return _try_research_fallback(statement, llm_client, result)

    return result


def _try_research_fallback(
    statement: str, llm_client: OpenAI, original_result: dict
) -> dict:
    """Attempt web research fallback. Returns original result on failure."""
    try:
        print("  [Spúšťam webový výskum...]\n", file=sys.stderr)
        return research_statement(statement, llm_client)
    except Exception as e:
        print(f"  Webový výskum zlyhal: {e}\n", file=sys.stderr)
        return original_result


def main():
    args = parse_args()

    # Initialize connections once
    qdrant = QdrantClient(host=EC2_IP, port=QDRANT_PORT)
    llm_client = get_openrouter_client()

    print(f"Overovač výrokov | Qdrant: {EC2_IP}:{QDRANT_PORT} | Prah: {args.threshold}")
    print("Zadajte politický výrok na overenie. Prázdny riadok alebo Ctrl+C na ukončenie.\n")

    while True:
        try:
            statement = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nUkončenie.")
            break

        if not statement:
            print("Ukončenie.")
            break

        try:
            result = verify_statement(
                statement, qdrant, llm_client, args.threshold,
                enable_research=not args.no_research,
            )
        except json.JSONDecodeError as e:
            print(f"Chyba: LLM vrátil neplatný JSON: {e}\n", file=sys.stderr)
            continue
        except requests.exceptions.ConnectionError:
            print(
                "Chyba: Nedá sa pripojiť k serveru (embedding/Qdrant).\n",
                file=sys.stderr,
            )
            continue
        except requests.exceptions.HTTPError as e:
            print(f"Chyba HTTP: {e}\n", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Neočakávaná chyba: {e}\n", file=sys.stderr)
            continue

        if args.json:
            print_json(result)
        else:
            print_human_readable(result)


if __name__ == "__main__":
    main()
