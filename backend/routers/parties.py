from fastapi import APIRouter, Query
from backend.models import PartySummary
from backend.data_loader import get_vyroky_df

router = APIRouter(prefix="/api", tags=["parties"])


@router.get("/parties", response_model=list[PartySummary])
def list_parties(
    sort_by: str = Query("total", pattern="^(total|politicka_strana)$"),
    sort_desc: bool = True,
):
    df = get_vyroky_df()
    df = df[df["Politická strana"] != ""]

    results = []
    for strana, group in df.groupby("Politická strana"):
        verdicts = group["Vyhodnotenie"].value_counts().to_dict()
        politicians_count = group["Meno"].nunique()
        results.append(
            PartySummary(
                politicka_strana=strana,
                total=len(group),
                verdicts=verdicts,
                politicians_count=politicians_count,
            )
        )

    key = "total" if sort_by == "total" else "politicka_strana"
    results.sort(
        key=lambda x: getattr(x, key),
        reverse=sort_desc if key == "total" else not sort_desc,
    )
    return results
