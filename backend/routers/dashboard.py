import hashlib

from fastapi import APIRouter, Query, HTTPException

from backend.models import DashboardStats, PartyStats, TopicStats, PoliticianStats
from backend.data_loader import get_vyroky_df

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

_VERDICT_MAP = {
    "Pravda": "true",
    "Nepravda": "false",
    "Zavádzajúce": "misleading",
    "Neoveriteľné": "uncheckable",
    "Neviem posúdiť": "uncheckable",
}


def _politician_id(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:12]


def _verdict_counts(series) -> dict[str, int]:
    """Count verdicts mapped to English keys from a Series of Slovak verdicts."""
    counts = {"true": 0, "false": 0, "misleading": 0, "uncheckable": 0}
    for v in series:
        eng = _VERDICT_MAP.get(str(v).strip(), "uncheckable")
        counts[eng] += 1
    return counts


def _truth_rate(counts: dict[str, int]) -> float:
    """% of checkable statements (true + false + misleading) that are true."""
    checkable = counts["true"] + counts["false"] + counts["misleading"]
    if checkable == 0:
        return 0.0
    return round(counts["true"] / checkable * 100, 1)


@router.get("/stats", response_model=DashboardStats)
def dashboard_stats():
    df = get_vyroky_df()
    total = len(df)
    breakdown = _verdict_counts(df["Vyhodnotenie"])
    truth = _truth_rate(breakdown)
    checkable = breakdown["true"] + breakdown["false"] + breakdown["misleading"]
    false_rate = round(breakdown["false"] / checkable * 100, 1) if checkable else 0.0

    # by party
    by_party = []
    for party, grp in df.groupby("Politická strana"):
        if not str(party).strip():
            continue
        vc = _verdict_counts(grp["Vyhodnotenie"])
        by_party.append(PartyStats(party=party, total=len(grp), **vc))
    by_party.sort(key=lambda p: p.total, reverse=True)

    # by topic
    by_topic = []
    for topic, grp in df.groupby("Oblast"):
        if not str(topic).strip():
            continue
        vc = _verdict_counts(grp["Vyhodnotenie"])
        by_topic.append(TopicStats(topic=topic, total=len(grp), **vc))
    by_topic.sort(key=lambda t: t.total, reverse=True)

    # by politician
    by_politician = []
    for name, grp in df.groupby("Meno"):
        if not str(name).strip():
            continue
        vc = _verdict_counts(grp["Vyhodnotenie"])
        party = grp["Politická strana"].mode()
        party = party.iloc[0] if len(party) > 0 else ""
        by_politician.append(
            PoliticianStats(
                id=_politician_id(name),
                name=name,
                party=party,
                total=len(grp),
                truthRate=_truth_rate(vc),
                **vc,
            )
        )
    by_politician.sort(key=lambda p: p.total, reverse=True)

    return DashboardStats(
        totalStatements=total,
        truthRate=truth,
        falseRate=false_rate,
        verdictBreakdown=breakdown,
        byParty=by_party,
        byTopic=by_topic,
        byPolitician=by_politician,
    )


@router.get("/party", response_model=list[PartyStats])
def dashboard_party(
    parties: str = Query(..., description="Comma-separated party names"),
):
    df = get_vyroky_df()
    requested = [p.strip() for p in parties.split(",") if p.strip()]
    results = []
    for party_name in requested:
        mask = df["Politická strana"].str.contains(party_name, case=False, na=False)
        grp = df[mask]
        if grp.empty:
            continue
        vc = _verdict_counts(grp["Vyhodnotenie"])
        # Use the most common exact party name from matches
        actual = grp["Politická strana"].mode()
        actual = actual.iloc[0] if len(actual) > 0 else party_name
        results.append(PartyStats(party=actual, total=len(grp), **vc))
    return results


@router.get("/politician/{politician_id}", response_model=PoliticianStats)
def dashboard_politician(politician_id: str):
    df = get_vyroky_df()

    for name, grp in df.groupby("Meno"):
        if _politician_id(name) == politician_id:
            vc = _verdict_counts(grp["Vyhodnotenie"])
            party = grp["Politická strana"].mode()
            party = party.iloc[0] if len(party) > 0 else ""
            return PoliticianStats(
                id=politician_id,
                name=name,
                party=party,
                total=len(grp),
                truthRate=_truth_rate(vc),
                **vc,
            )

    raise HTTPException(status_code=404, detail="Politician not found")
