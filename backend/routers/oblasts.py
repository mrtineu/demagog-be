from fastapi import APIRouter
from backend.models import OblastSummary, VyrokItem
from backend.data_loader import get_vyroky_df

router = APIRouter(prefix="/api", tags=["oblasts"])


@router.get("/oblasts", response_model=list[OblastSummary])
def list_oblasts():
    df = get_vyroky_df()
    df = df[df["Oblast"] != ""]

    results = []
    for oblast, group in df.groupby("Oblast"):
        results.append(OblastSummary(oblast=oblast, total=len(group)))

    results.sort(key=lambda x: x.total, reverse=True)
    return results


@router.get("/recent", response_model=list[VyrokItem], tags=["trending"])
def recent_vyroky():
    """Most recent 20 fact-checked statements."""
    df = get_vyroky_df()
    df = df[df["Dátum"] != ""]
    df = df[df["Dátum"] != "0000-00-00"]
    recent = df.sort_values("Dátum", ascending=False).head(20)

    return [
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
