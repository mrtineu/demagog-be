import json
import logging

from fastapi import APIRouter, HTTPException

from backend.models import (
    ResearchRequest,
    VerifyRequest,
    VerifyResponse,
    VerifySource,
    WebSource,
)
from backend.services.llm_client import get_openrouter_client
from backend.services.research_service import research_statement
from backend.services.verification_service import verify_statement
from shared.verdicts import VERDICT_CORRECTNESS

router = APIRouter(prefix="/api", tags=["verify"])
logger = logging.getLogger(__name__)


# ── Converters ─────────────────────────────────────────────────────────


def _dict_to_verify_response(
    result: dict, research_available: bool = False
) -> VerifyResponse:
    """Convert a verification service dict to a VerifyResponse model."""
    zdroj = None
    raw_zdroj = result.get("zdroj")
    if raw_zdroj and isinstance(raw_zdroj, dict):
        zdroj = VerifySource(**raw_zdroj)

    verdikt = result.get("verdikt", "Nedostatok dát")
    return VerifyResponse(
        vstupny_vyrok=result["vstupny_vyrok"],
        status=result["status"],
        verdikt=verdikt,
        verdikt_label=result.get("verdikt_label", verdikt),
        je_pravda=VERDICT_CORRECTNESS.get(verdikt),
        odovodnenie_llm=result.get("odovodnenie_llm", ""),
        zdrojovy_vyrok=result.get("zdrojovy_vyrok"),
        zdroj=zdroj,
        pouzity_prah=result.get("pouzity_prah"),
        pocet_nad_prahom=result.get("pocet_nad_prahom", 0),
        pocet_celkom=result.get("pocet_celkom", 0),
        research_available=research_available,
    )


def _dict_to_research_response(research: dict) -> VerifyResponse:
    """Convert a research service dict to a VerifyResponse model."""
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

    verdikt = research.get("verdikt", "Neoveriteľné")
    return VerifyResponse(
        vstupny_vyrok=research["vstupny_vyrok"],
        status=research.get("status", "webovy_vyskum"),
        verdikt=verdikt,
        verdikt_label=research.get("verdikt_label", ""),
        je_pravda=VERDICT_CORRECTNESS.get(verdikt),
        odovodnenie_llm=research.get("odovodnenie_llm", ""),
        zdrojovy_vyrok=None,
        zdroj=None,
        pouzity_prah=None,
        pocet_nad_prahom=0,
        pocet_celkom=0,
        research_available=False,
        web_research_used=True,
        typ_overenia="webovy_vyskum",
        istota=research.get("istota"),
        webove_zdroje=web_sources if web_sources else None,
        pocet_najdenych_zdrojov=research.get("pocet_najdenych_zdrojov"),
        pocet_podpornych_zdrojov=research.get("pocet_podpornych_zdrojov"),
        protirecie=research.get("protirecie"),
        safeguard_override=research.get("safeguard_override"),
    )


# ── Endpoints ──────────────────────────────────────────────────────────


@router.post("/verify", response_model=VerifyResponse)
def verify(body: VerifyRequest):
    """Verify a political statement against the Demagog.sk database."""
    client = get_openrouter_client()

    try:
        result = verify_statement(
            statement=body.vyrok,
            llm_client=client,
            threshold=body.threshold,
            top_k=body.top_k,
        )
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502, detail=f"LLM returned invalid JSON: {exc}"
        )

    research_available = result.get("verdikt") == "Nedostatok dát"
    return _dict_to_verify_response(result, research_available=research_available)


@router.post("/research", response_model=VerifyResponse)
def research(body: ResearchRequest):
    """Run web research for a statement that had no DB match.

    Called explicitly by the user (via a frontend button) after
    /api/verify returns research_available=True.
    """
    client = get_openrouter_client()

    try:
        result = research_statement(body.vyrok, client)
    except Exception as exc:
        logger.error("Web research failed: %s", exc)
        return VerifyResponse(
            vstupny_vyrok=body.vyrok,
            status="bez_zhody",
            verdikt="Nedostatok dát",
            verdikt_label="NEDOSTATOK DÁT",
            je_pravda=None,
            odovodnenie_llm=f"Webový výskum zlyhal: {exc}",
            zdrojovy_vyrok=None,
            zdroj=None,
            pouzity_prah=body.threshold_used,
            pocet_nad_prahom=0,
            pocet_celkom=0,
            research_available=False,
            web_research_used=True,
        )

    return _dict_to_research_response(result)
