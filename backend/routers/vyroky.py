from fastapi import APIRouter, Query
from backend.models import VyrokItem, PaginatedVyroky
from backend.data_loader import get_vyroky_df

router = APIRouter(prefix="/api", tags=["vyroky"])


@router.get("/vyroky", response_model=PaginatedVyroky)
def list_vyroky(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    meno: str | None = None,
    strana: str | None = None,
    vyhodnotenie: str | None = None,
    oblast: str | None = None,
    datum_od: str | None = None,
    datum_do: str | None = None,
):
    df = get_vyroky_df()

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
