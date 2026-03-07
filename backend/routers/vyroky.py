from enum import Enum
import logging

from fastapi import APIRouter, Query, HTTPException
from backend.models import VyrokItem, VyrokCreate, PaginatedVyroky
from backend.data_loader import get_vyroky_df, append_vyrok

router = APIRouter(prefix="/api", tags=["vyroky"])

logger = logging.getLogger(__name__)


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

@router.post("/vyroky", response_model=VyrokItem, status_code=201)
def create_vyrok(body: VyrokCreate):
    row = {
        "Výrok": body.vyrok,
        "Vyhodnotenie": body.vyhodnotenie,
        "Odôvodnenie": body.odovodnenie,
        "Oblast": body.oblast,
        "Dátum": body.datum,
        "Meno": body.meno,
        "Politická strana": body.politicka_strana,
    }

    append_vyrok(row)

    # Best-effort upsert into Qdrant so the new statement is searchable
    try:
        from backend.qdrant_service import upsert_vyrok
        upsert_vyrok(row)
    except Exception:
        logger.warning("Failed to upsert new výrok into Qdrant", exc_info=True)

    return VyrokItem(
        vyrok=body.vyrok,
        vyhodnotenie=body.vyhodnotenie,
        odovodnenie=body.odovodnenie,
        oblast=body.oblast,
        datum=body.datum,
        meno=body.meno,
        politicka_strana=body.politicka_strana,
    )
