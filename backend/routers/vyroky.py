from enum import Enum

from fastapi import APIRouter, Query
from backend.models import VyrokItem, PaginatedVyroky
from backend.data_loader import get_vyroky_df

router = APIRouter(prefix="/api", tags=["vyroky"])


class VyrokySortBy(str, Enum):
    datum = "datum"
    meno = "meno"
    strana = "strana"
    vyhodnotenie = "vyhodnotenie"


_SORT_COLUMN = {
    VyrokySortBy.datum: "Dátum",
    VyrokySortBy.meno: "Meno",
    VyrokySortBy.strana: "Politická strana",
    VyrokySortBy.vyhodnotenie: "Vyhodnotenie",
}


@router.get("/vyroky", response_model=PaginatedVyroky)
def list_vyroky(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    q: str | None = None,
    meno: str | None = None,
    strana: str | None = None,
    vyhodnotenie: str | None = None,
    oblast: str | None = None,
    datum_od: str | None = None,
    datum_do: str | None = None,
    sort_by: VyrokySortBy = VyrokySortBy.datum,
    sort_desc: bool = True,
):
    df = get_vyroky_df()

    if q:
        mask = (
            df["Výrok"].str.contains(q, case=False, na=False)
            | df["Odôvodnenie"].str.contains(q, case=False, na=False)
        )
        df = df[mask]
    if meno:
        df = df[df["Meno"].str.contains(meno, case=False, na=False)]
    if strana:
        df = df[df["Politická strana"].str.contains(strana, case=False, na=False)]
    if vyhodnotenie:
        df = df[df["Vyhodnotenie"].str.lower() == vyhodnotenie.lower()]
    if oblast:
        df = df[df["Oblast"].str.contains(oblast, case=False, na=False)]
    if datum_od:
        df = df[df["Dátum"] >= datum_od]
    if datum_do:
        df = df[df["Dátum"] <= datum_do]

    col = _SORT_COLUMN[sort_by]
    df = df.sort_values(col, ascending=not sort_desc, na_position="last")

    total = len(df)
    start = (page - 1) * page_size
    page_df = df.iloc[start : start + page_size]

    items = [
        VyrokItem(
            vyrok=row["Výrok"],
            vyhodnotenie=row["Vyhodnotenie"],
            odovodnenie=row["Odôvodnenie"],
            oblast=row["Oblast"],
            datum=row["Dátum"],
            meno=row["Meno"],
            politicka_strana=row["Politická strana"],
        )
        for _, row in page_df.iterrows()
    ]

    return PaginatedVyroky(items=items, total=total, page=page, page_size=page_size)
