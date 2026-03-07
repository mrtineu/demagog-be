from fastapi import APIRouter, HTTPException, Query
from backend.models import PoliticianSummary, PoliticianDetail, VyrokItem
from backend.data_loader import get_vyroky_df

router = APIRouter(prefix="/api", tags=["politicians"])


@router.get("/politicians", response_model=list[PoliticianSummary])
def list_politicians(
    strana: str | None = None,
    sort_by: str = Query("total", pattern="^(total|meno)$"),
    sort_desc: bool = True,
):
    df = get_vyroky_df()
    df = df[df["Meno"] != ""]

    if strana:
        df = df[df["Politická strana"].str.contains(strana, case=False, na=False)]

    results = []
    for meno, group in df.groupby("Meno"):
        verdicts = group["Vyhodnotenie"].value_counts().to_dict()
        strana_val = group["Politická strana"].mode()
        results.append(
            PoliticianSummary(
                meno=meno,
                politicka_strana=strana_val.iloc[0] if len(strana_val) > 0 else "",
                total=len(group),
                verdicts=verdicts,
            )
        )

    key = "total" if sort_by == "total" else "meno"
    results.sort(
        key=lambda x: getattr(x, key),
        reverse=sort_desc if key == "total" else not sort_desc,
    )
    return results


@router.get("/politicians/{name}", response_model=PoliticianDetail)
def get_politician(name: str, recent_limit: int = Query(10, ge=1, le=50)):
    df = get_vyroky_df()
    politician_df = df[df["Meno"].str.lower() == name.lower()]

    if politician_df.empty:
        raise HTTPException(status_code=404, detail=f"Politician '{name}' not found")

    verdicts = politician_df["Vyhodnotenie"].value_counts().to_dict()
    oblasts = (
        politician_df["Oblast"][politician_df["Oblast"] != ""]
        .value_counts()
        .to_dict()
    )
    strana = politician_df["Politická strana"].mode()

    recent = politician_df.sort_values("Dátum", ascending=False).head(recent_limit)
    recent_items = [
        VyrokItem(
            vyrok=row["Výrok"],
            vyhodnotenie=row["Vyhodnotenie"],
            odovodnenie=row["Odôvodnenie"],
            oblast=row["Oblast"],
            datum=row["Dátum"],
            meno=row["Meno"],
            politicka_strana=row["Politická strana"],
        )
        for _, row in recent.iterrows()
    ]

    return PoliticianDetail(
        meno=politician_df["Meno"].iloc[0],
        politicka_strana=strana.iloc[0] if len(strana) > 0 else "",
        total=len(politician_df),
        verdicts=verdicts,
        oblasts=oblasts,
        recent_vyroky=recent_items,
    )
