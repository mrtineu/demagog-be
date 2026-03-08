import json
import logging

from fastapi import APIRouter, HTTPException

from backend.models import VerifyRequest, VerifyResponse, VerifySource, WebSource
from backend.qdrant_service import get_qdrant_client, embed
from backend.config import COLLECTION_NAME, LLM_MODEL, LLM_TEMPERATURE
from backend.services.llm_client import get_openrouter_client
from backend.services.research_service import research_statement
from shared.verdicts import VERDICT_LABEL, VERDICT_CORRECTNESS
from shared.prompts import VERIFY_SYSTEM_PROMPT as SYSTEM_PROMPT

router = APIRouter(prefix="/api", tags=["verify"])
logger = logging.getLogger(__name__)

# Incomplete-statement detection
MIN_WORD_COUNT = 4
MIN_LENGTH_RATIO = 0.5

# ── Incomplete-statement detection ────────────────────────────────────

def _is_incomplete_statement(statement: str, db_results: list[dict]) -> tuple[bool, str]:
    """Return (True, reason) if the statement is obviously fragmentary."""
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


# ── Helpers ─────────────────────────────────────────────────────────────

def _search_similar(query_text: str, top_k: int) -> list[dict]:
    vector = embed(query_text)
    client = get_qdrant_client()
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
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
        for hit in response.points
    ]


def _build_user_message(input_statement: str, db_results: list[dict]) -> str:
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


def _call_llm(input_statement: str, db_results: list[dict]) -> dict:
    client = get_openrouter_client()
    user_message = _build_user_message(input_statement, db_results)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()

    return json.loads(raw)


def _build_result(
    input_statement: str,
    llm_resp: dict,
    above: list[dict],
    all_results: list[dict],
    threshold: float,
) -> VerifyResponse:
    zdroj = None
    idx = llm_resp.get("index_zhody")
    if llm_resp.get("zhoda") and idx is not None and 1 <= idx <= len(above):
        m = above[idx - 1]
        zdroj = VerifySource(
            vyrok=m["vyrok"],
            vyhodnotenie=m["vyhodnotenie"],
            odovodnenie=m["odovodnenie"],
            oblast=m["oblast"],
            datum=m["datum"],
            meno=m["meno"],
            politicka_strana=m["politicka_strana"],
            skore_podobnosti=m["score"],
        )

    verdikt = llm_resp.get("verdikt", "Nedostatok dát")
    return VerifyResponse(
        vstupny_vyrok=input_statement,
        status="zhoda" if llm_resp.get("zhoda") else "bez_zhody",
        verdikt=verdikt,
        verdikt_label=VERDICT_LABEL.get(verdikt, verdikt),
        je_pravda=VERDICT_CORRECTNESS.get(verdikt),
        odovodnenie_llm=llm_resp.get("odovodnenie_llm", ""),
        zdrojovy_vyrok=llm_resp.get("zdrojovy_vyrok"),
        zdroj=zdroj,
        pouzity_prah=threshold,
        pocet_nad_prahom=len(above),
        pocet_celkom=len(all_results),
    )


def _build_no_data(
    input_statement: str,
    all_results: list[dict],
    threshold: float,
) -> VerifyResponse:
    top_score = all_results[0]["score"] if all_results else 0.0
    return VerifyResponse(
        vstupny_vyrok=input_statement,
        status="bez_zhody",
        verdikt="Nedostatok dát",
        verdikt_label="NEDOSTATOK DÁT",
        je_pravda=None,
        odovodnenie_llm=(
            f"Žiadny výsledok z databázy nepresiahol prah podobnosti {threshold}. "
            f"Najvyššie skóre: {top_score:.4f}."
        ),
        zdrojovy_vyrok=None,
        zdroj=None,
        pouzity_prah=threshold,
        pocet_nad_prahom=0,
        pocet_celkom=len(all_results),
    )


# ── Research fallback helpers ────────────────────────────────────────────

def _build_research_response(research: dict, threshold: float | None) -> VerifyResponse:
    """Map research_agent output dict to VerifyResponse."""
    raw_sources = research.get("webove_zdroje") or []
    web_sources = [
        WebSource(
            url=s.get("url", ""),
            nazov=s.get("nazov", ""),
            relevantny_citat=s.get("relevantny_citat", ""),
            typ_zdroja=s.get("typ_zdroja", ""),
        )
        for s in raw_sources
        if isinstance(s, dict)
    ]
    return VerifyResponse(
        vstupny_vyrok=research["vstupny_vyrok"],
        status=research.get("status", "webovy_vyskum"),
        verdikt=research.get("verdikt", "Neoveriteľné"),
        verdikt_label=research.get("verdikt_label", ""),
        je_pravda=VERDICT_CORRECTNESS.get(research.get("verdikt", "")),
        odovodnenie_llm=research.get("odovodnenie_llm", ""),
        zdrojovy_vyrok=None,
        zdroj=None,
        pouzity_prah=None,
        pocet_nad_prahom=0,
        pocet_celkom=0,
        web_research_used=True,
        typ_overenia="webovy_vyskum",
        istota=research.get("istota"),
        webove_zdroje=web_sources if web_sources else None,
        pocet_najdenych_zdrojov=research.get("pocet_najdenych_zdrojov"),
        pocet_podpornych_zdrojov=research.get("pocet_podpornych_zdrojov"),
        protirecie=research.get("protirecie"),
        safeguard_override=research.get("safeguard_override"),
    )


def _research_fallback(statement: str, threshold: float) -> VerifyResponse:
    """Call the web research agent. On any failure return a safe bez_zhody response."""
    try:
        research_dict = research_statement(statement, get_openrouter_client())
        return _build_research_response(research_dict, threshold)
    except Exception as exc:
        logger.error("Web research fallback failed: %s", exc)
        return VerifyResponse(
            vstupny_vyrok=statement,
            status="bez_zhody",
            verdikt="Nedostatok dát",
            verdikt_label="NEDOSTATOK DÁT",
            je_pravda=None,
            odovodnenie_llm=(
                f"Databáza neobsahuje zhodu a webový výskum zlyhal: {exc}"
            ),
            zdrojovy_vyrok=None,
            zdroj=None,
            pouzity_prah=threshold,
            pocet_nad_prahom=0,
            pocet_celkom=0,
        )


# ── Endpoint ────────────────────────────────────────────────────────────

@router.post("/verify", response_model=VerifyResponse)
def verify_statement(body: VerifyRequest):
    """Verify a political statement against the Demagog.sk database."""
    all_results = _search_similar(body.vyrok, body.top_k)

    if not all_results:
        if body.enable_research:
            return _research_fallback(body.vyrok, body.threshold)
        return _build_no_data(body.vyrok, all_results, body.threshold)

    above = [r for r in all_results if r["score"] >= body.threshold]

    if not above:
        if body.enable_research:
            return _research_fallback(body.vyrok, body.threshold)
        return _build_no_data(body.vyrok, all_results, body.threshold)

    # Step 2.5: Reject incomplete / fragmentary input (no research for fragments)
    incomplete, reason = _is_incomplete_statement(body.vyrok, above or all_results)
    if incomplete:
        return VerifyResponse(
            vstupny_vyrok=body.vyrok,
            status="bez_zhody",
            verdikt="Nedostatok dát",
            verdikt_label="NEDOSTATOK DÁT",
            je_pravda=None,
            odovodnenie_llm=reason,
            zdrojovy_vyrok=None,
            zdroj=None,
            pouzity_prah=body.threshold,
            pocet_nad_prahom=len(above),
            pocet_celkom=len(all_results),
        )

    try:
        llm_resp = _call_llm(body.vyrok, above)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {exc}")

    result = _build_result(body.vyrok, llm_resp, above, all_results, body.threshold)

    # If LLM found no match in DB, try web research fallback
    if body.enable_research and result.verdikt == "Nedostatok dát":
        return _research_fallback(body.vyrok, body.threshold)

    return result
