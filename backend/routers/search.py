from fastapi import APIRouter, HTTPException, Query
from backend.models import SearchResult
from backend.qdrant_service import search_similar

router = APIRouter(prefix="/api", tags=["search"])

VERDICT_LABEL = {
    "Pravda": "PRAVDA",
    "Nepravda": "NEPRAVDA",
    "Zavádzajúce": "ZAVÁDZAJÚCE",
    "Neoveriteľné": "NEOVERITEĽNÉ",
}


@router.get("/search", response_model=list[SearchResult])
def search(
    query: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=50),
):
    try:
        results = search_similar(query, top_k)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Search service error: {e}")

    return [
        SearchResult(
            vyrok=r["vyrok"],
            vyhodnotenie=r["vyhodnotenie"],
            vyhodnotenie_label=VERDICT_LABEL.get(r["vyhodnotenie"], r["vyhodnotenie"]),
            odovodnenie=r["odovodnenie"],
            oblast=r["oblast"],
            datum=r["datum"],
            meno=r["meno"],
            politicka_strana=r["politicka_strana"],
            score=r["score"],
        )
        for r in results
    ]
