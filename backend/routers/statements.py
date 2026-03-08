import hashlib

from fastapi import APIRouter, Query, HTTPException

from backend.models import Statement
from backend.data_loader import get_vyroky_df
from shared.verdicts import VERDICT_MAP as _VERDICT_MAP

router = APIRouter(prefix="/api", tags=["statements"])

# Reverse mapping for filtering
_VERDICT_MAP_REV = {v: k for k, v in _VERDICT_MAP.items()}


def _row_id(row) -> str:
    """Deterministic ID from row content."""
    key = f"{row['Meno']}|{row['Výrok']}|{row['Dátum']}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _row_to_statement(row) -> Statement:
    verdict_sk = str(row["Vyhodnotenie"]).strip()
    return Statement(
        id=_row_id(row),
        politicianName=row["Meno"],
        politicianParty=row["Politická strana"],
        statementText=row["Výrok"],
        verdict=_VERDICT_MAP.get(verdict_sk, "uncheckable"),
        date=row["Dátum"],
        explanation=row["Odôvodnenie"],
        topic=row["Oblast"],
    )


@router.get("/statements", response_model=list[Statement])
def list_statements(
    q: str | None = None,
    party: str | None = None,
    verdict: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
):
    df = get_vyroky_df()

    if q:
        mask = (
            df["Výrok"].str.contains(q, case=False, na=False)
            | df["Odôvodnenie"].str.contains(q, case=False, na=False)
        )
        df = df[mask]
    if party:
        df = df[df["Politická strana"].str.contains(party, case=False, na=False)]
    if verdict:
        sk_verdict = _VERDICT_MAP_REV.get(verdict.lower())
        if sk_verdict:
            df = df[df["Vyhodnotenie"].str.lower() == sk_verdict.lower()]
    if date_from:
        df = df[df["Dátum"] >= date_from]
    if date_to:
        df = df[df["Dátum"] <= date_to]

    df = df.sort_values("Dátum", ascending=False, na_position="last")

    return [_row_to_statement(row) for _, row in df.iterrows()]


@router.get("/statements/{statement_id}", response_model=Statement)
def get_statement(statement_id: str):
    df = get_vyroky_df()

    for _, row in df.iterrows():
        if _row_id(row) == statement_id:
            return _row_to_statement(row)

    raise HTTPException(status_code=404, detail="Statement not found")
