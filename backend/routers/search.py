from fastapi import APIRouter, HTTPException, Query
from backend.data_loader import get_vyroky_df
from backend.models import ExactMatch, SearchResult, PaginatedSearchResults
from backend.qdrant_service import search_similar

router = APIRouter(prefix="/api", tags=["search"])

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


def _find_exact_match(query: str) -> ExactMatch | None:
    """Case-insensitive exact match against the CSV."""
    df = get_vyroky_df()
    needle = query.strip().lower()
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


@router.get("/search", response_model=PaginatedSearchResults)
def search(
    query: str = Query(..., min_length=1),
    top_k: int = Query(50, ge=1, le=200),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
):
    exact = _find_exact_match(query)
    exact_text_lower = query.strip().lower()

    try:
        results = search_similar(query, top_k)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Search service error: {e}")

    items = [
        SearchResult(
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
        for r in results
        # exclude the exact match row from similar results to avoid duplication
        if r["vyrok"].strip().lower() != exact_text_lower or exact is None
    ]

    total = len(items)
    start = (page - 1) * page_size
    page_items = items[start : start + page_size]

    return PaginatedSearchResults(exact_match=exact, items=page_items, total=total, page=page, page_size=page_size)
