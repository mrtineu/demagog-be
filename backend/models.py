from pydantic import BaseModel, Field


# --- Search ---

class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)


class SearchResult(BaseModel):
    vyrok: str
    vyhodnotenie: str
    vyhodnotenie_label: str
    odovodnenie: str
    oblast: str
    datum: str
    meno: str
    politicka_strana: str
    score: float


# --- Vyroky (statements) ---

class VyrokItem(BaseModel):
    vyrok: str
    vyhodnotenie: str
    odovodnenie: str
    oblast: str
    datum: str
    meno: str
    politicka_strana: str


class PaginatedVyroky(BaseModel):
    items: list[VyrokItem]
    total: int
    page: int
    page_size: int


# --- Clanky (articles) ---

class ClanokItem(BaseModel):
    datum: str
    autor: str
    text: str


class PaginatedClanky(BaseModel):
    items: list[ClanokItem]
    total: int
    page: int
    page_size: int


# --- Stats ---

class StatsResponse(BaseModel):
    total_vyroky: int
    total_clanky: int
    verdicts: dict[str, int]
    parties: dict[str, int]
    politicians: dict[str, int]


# --- Politicians ---

class PoliticianSummary(BaseModel):
    meno: str
    politicka_strana: str
    total: int
    verdicts: dict[str, int]


class PoliticianDetail(BaseModel):
    meno: str
    politicka_strana: str
    total: int
    verdicts: dict[str, int]
    oblasts: dict[str, int]
    recent_vyroky: list[VyrokItem]


# --- Parties ---

class PartySummary(BaseModel):
    politicka_strana: str
    total: int
    verdicts: dict[str, int]
    politicians_count: int


# --- Oblasts ---

class OblastSummary(BaseModel):
    oblast: str
    total: int
