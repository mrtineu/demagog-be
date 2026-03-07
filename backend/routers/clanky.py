from fastapi import APIRouter, Query
from backend.models import ClanokItem, PaginatedClanky
from backend.data_loader import get_clanky_df

router = APIRouter(prefix="/api", tags=["clanky"])


@router.get("/clanky", response_model=PaginatedClanky)
def list_clanky(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    autor: str | None = None,
):
    df = get_clanky_df()

    if autor:
        df = df[df["Autor"].str.contains(autor, case=False, na=False)]

    total = len(df)
    start = (page - 1) * page_size
    page_df = df.iloc[start : start + page_size]

    items = [
        ClanokItem(
            datum=row["Dátum"],
            autor=row["Autor"],
            text=row["Text"],
        )
        for _, row in page_df.iterrows()
    ]

    return PaginatedClanky(items=items, total=total, page=page, page_size=page_size)
