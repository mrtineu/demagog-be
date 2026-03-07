from fastapi import APIRouter, HTTPException
from backend.models import SearchRequest, SearchResult
from backend.qdrant_service import search_similar

router = APIRouter(prefix="/api", tags=["search"])

VERDICT_LABEL = {
    "Pravda": "PRAVDA",
    "Nepravda": "NEPRAVDA",
    "Zavádzajúce": "ZAVÁDZAJÚCE",
    "Neoveriteľné": "NEOVERITEĽNÉ",
}


@router.post("/search", response_model=list[SearchResult])
def search(req: SearchRequest):
    try:
        results = search_similar(req.query, req.top_k)
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
