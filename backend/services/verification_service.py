"""Statement verification service adapted from Client/verify.py.

Reuses backend.qdrant_service for embedding and search instead of
duplicating that logic. Uses backend.config for all settings.
"""

import json
import logging

from openai import OpenAI

from backend.config import LLM_MODEL, LLM_TEMPERATURE
from backend.qdrant_service import search_similar
from shared.verdicts import VERDICT_LABEL
from shared.prompts import VERIFY_SYSTEM_PROMPT as SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# -- Constants --

TOP_K = 5
DEFAULT_THRESHOLD = 0.6
MIN_WORD_COUNT = 4
MIN_LENGTH_RATIO = 0.2


# -- Helper functions --


def filter_by_threshold(results: list[dict], threshold: float) -> list[dict]:
    """Keep only results with similarity score >= threshold."""
    return [r for r in results if r["score"] >= threshold]


def is_incomplete_statement(statement: str, db_results: list[dict]) -> tuple[bool, str]:
    """Detect obviously incomplete or fragmentary input statements."""
    words = statement.split()

    if len(words) < MIN_WORD_COUNT:
        return True, (
            f"Vstupný výrok obsahuje iba {len(words)} slov(á). "
            "Úplné faktické tvrdenie vyžaduje aspoň podmět, prísudok a predmět."
        )

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


# -- Result assembly --


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
        "all_results": all_raw_results,
    }

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
        "all_results": all_raw_results,
    }


# -- Main verification pipeline --


def verify_statement(
    statement: str,
    llm_client: OpenAI,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = TOP_K,
) -> dict:
    """Full verification pipeline for a single statement.

    Args:
        statement: The political statement to verify.
        llm_client: OpenRouter client for LLM calls.
        threshold: Similarity score threshold for DB matching.
        top_k: Number of similar results to retrieve from the database.

    Returns:
        Verification result dict with verdict, reasoning, and sources.
    """
    # Step 1: Search Qdrant
    all_results = search_similar(statement, top_k=top_k)

    if not all_results:
        return build_no_data_result(statement, all_results, threshold)

    # Step 2: Filter by threshold
    above_threshold = filter_by_threshold(all_results, threshold)

    if not above_threshold:
        return build_no_data_result(statement, all_results, threshold)

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
            "all_results": all_results,
        }

    # Step 3: LLM evaluation
    llm_response = call_llm(llm_client, statement, above_threshold)

    # Step 4: Assemble result
    return build_verification_result(
        statement, llm_response, above_threshold, all_results, threshold
    )
