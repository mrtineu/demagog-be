from fastapi import APIRouter
from backend.models import StatsResponse
from backend.data_loader import get_vyroky_df, get_clanky_df

router = APIRouter(prefix="/api", tags=["stats"])


@router.get("/stats", response_model=StatsResponse)
def get_stats():
    vyroky_df = get_vyroky_df()
    clanky_df = get_clanky_df()

    verdicts = (
        vyroky_df["Vyhodnotenie"]
        .value_counts()
        .to_dict()
    )
    parties = (
        vyroky_df["Politická strana"]
        [vyroky_df["Politická strana"] != ""]
        .value_counts()
        .to_dict()
    )
    politicians = (
        vyroky_df["Meno"]
        [vyroky_df["Meno"] != ""]
        .value_counts()
        .to_dict()
    )

    return StatsResponse(
        total_vyroky=len(vyroky_df),
        total_clanky=len(clanky_df),
        verdicts=verdicts,
        parties=parties,
        politicians=politicians,
    )
