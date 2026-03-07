from fastapi import APIRouter, HTTPException

from backend.data_loader import get_vyroky_df
from backend.models import ExactMatch, LookupRequest, LookupResponse, SimilarStatement
from backend.qdrant_service import search_similar

router = APIRouter(prefix="/api", tags=["lookup"])

VERDICT_LABEL = {
    "Pravda": "PRAVDA",
    "Nepravda": "NEPRAVDA",
    "Zavádzajúce": "ZAVÁDZAJÚCE",
    "Neoveriteľné": "NEOVERITEĽNÉ",
}

VERDICT_CORRECTNESS: dict[str, bool | None] = {
    "Pravda": True,
    "Nepravda": False,
    "Zavádzajúce": False,
    "Neoveriteľné": None,
}


def _exact_match(vyrok: str) -> ExactMatch | None:
    """Check the CSV for a word-for-word match (case-insensitive, stripped)."""
    df = get_vyroky_df()
    needle = vyrok.strip().lower()
    mask = df["Výrok"].str.strip().str.lower() == needle
    matches = df[mask]
    if matches.empty:
        return None
    row = matches.iloc[0]
    vyhodnotenie = str(row.get("Vyhodnotenie", ""))
    return ExactMatch(
        vyrok=str(row.get("Výrok", "")),
        vyhodnotenie=vyhodnotenie,
        vyhodnotenie_label=VERDICT_LABEL.get(vyhodnotenie, vyhodnotenie),
        je_pravda=VERDICT_CORRECTNESS.get(vyhodnotenie),
        odovodnenie=str(row.get("Odôvodnenie", "")),
        oblast=str(row.get("Oblast", "")),
        datum=str(row.get("Dátum", "")),
        meno=str(row.get("Meno", "")),
        politicka_strana=str(row.get("Politická strana", "")),
    )


@router.post("/lookup", response_model=LookupResponse)
def lookup(body: LookupRequest):
    """
    Return an exact word-for-word match (if one exists) plus an array of
    semantically similar statements from the database.
    """
    exact = _exact_match(body.vyrok)

    try:
        raw_similar = search_similar(body.vyrok, body.top_k)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Search service error: {e}")

    # Exclude the exact match from the similar list to avoid duplication
    exact_text_lower = body.vyrok.strip().lower()
    similar = [
        SimilarStatement(
            vyrok=r["vyrok"],
            vyhodnotenie=r["vyhodnotenie"],
            vyhodnotenie_label=VERDICT_LABEL.get(r["vyhodnotenie"], r["vyhodnotenie"]),
            je_pravda=VERDICT_CORRECTNESS.get(r["vyhodnotenie"]),
            odovodnenie=r["odovodnenie"],
            oblast=r["oblast"],
            datum=r["datum"],
            meno=r["meno"],
            politicka_strana=r["politicka_strana"],
            score=r["score"],
        )
        for r in raw_similar
        if r["vyrok"].strip().lower() != exact_text_lower
    ]

    return LookupResponse(exact_match=exact, similar=similar)
