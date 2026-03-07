from enum import Enum

from fastapi import APIRouter, Query
from backend.models import ClanokItem, PaginatedClanky
from backend.data_loader import get_clanky_df

router = APIRouter(prefix="/api", tags=["clanky"])


class ClankySortBy(str, Enum):
    datum = "datum"
    autor = "autor"


_SORT_COLUMN = {
    ClankySortBy.datum: "Dátum",
    ClankySortBy.autor: "Autor",
}


@router.get("/clanky", response_model=PaginatedClanky)
def list_clanky(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    q: str | None = None,
    autor: str | None = None,
    sort_by: ClankySortBy = ClankySortBy.datum,
    sort_desc: bool = True,
):
    df = get_clanky_df()

    if q:
        df = df[df["Text"].str.contains(q, case=False, na=False)]
    if autor:
        df = df[df["Autor"].str.contains(autor, case=False, na=False)]

    col = _SORT_COLUMN[sort_by]
    df = df.sort_values(col, ascending=not sort_desc, na_position="last")

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
