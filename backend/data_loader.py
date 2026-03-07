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


def get_clanky_df() -> pd.DataFrame:
    if _clanky_df is None:
        load_dataframes()
    return _clanky_df
