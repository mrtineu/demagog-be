import pandas as pd
from backend.config import VYROKY_CSV_PATH, CLANKY_CSV_PATH

_vyroky_df: pd.DataFrame | None = None
_clanky_df: pd.DataFrame | None = None


def load_dataframes():
    """Load both CSV files into memory. Called once at startup."""
    global _vyroky_df, _clanky_df
    _vyroky_df = pd.read_csv(VYROKY_CSV_PATH, delimiter=";").fillna("")
    _clanky_df = pd.read_csv(CLANKY_CSV_PATH, delimiter=";").fillna("")


def get_vyroky_df() -> pd.DataFrame:
    if _vyroky_df is None:
        load_dataframes()
    return _vyroky_df


def append_vyrok(row: dict) -> None:
    """Append a single statement row to the in-memory DataFrame and CSV."""
    global _vyroky_df
    if _vyroky_df is None:
        load_dataframes()
    new_row = pd.DataFrame([row])
    _vyroky_df = pd.concat([_vyroky_df, new_row], ignore_index=True)
    _vyroky_df.to_csv(VYROKY_CSV_PATH, sep=";", index=False)


def get_clanky_df() -> pd.DataFrame:
    if _clanky_df is None:
        load_dataframes()
    return _clanky_df
