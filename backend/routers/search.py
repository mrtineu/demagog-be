from fastapi import APIRouter, HTTPException, Query
from backend.models import SearchResult, PaginatedSearchResults
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


@router.get("/search", response_model=PaginatedSearchResults)
def search(
    query: str = Query(..., min_length=1),
    top_k: int = Query(50, ge=1, le=200),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
):
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
    ]

    total = len(items)
    start = (page - 1) * page_size
    page_items = items[start : start + page_size]

    return PaginatedSearchResults(items=page_items, total=total, page=page, page_size=page_size)
