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

class VyrokCreate(BaseModel):
    vyrok: str
    vyhodnotenie: str
    odovodnenie: str = ""
    oblast: str = ""
    datum: str
    meno: str
    politicka_strana: str = ""


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


# --- Verify ---

class VerifyRequest(BaseModel):
    vyrok: str
    threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    top_k: int = Field(default=5, ge=1, le=50)


class VerifySource(BaseModel):
    vyrok: str
    vyhodnotenie: str
    odovodnenie: str
    oblast: str
    datum: str
    meno: str
    politicka_strana: str
    skore_podobnosti: float


class VerifyResponse(BaseModel):
    vstupny_vyrok: str
    status: str
    verdikt: str
    verdikt_label: str
    odovodnenie_llm: str
    zdrojovy_vyrok: str | None
    zdroj: VerifySource | None
    pouzity_prah: float
    pocet_nad_prahom: int
    pocet_celkom: int


# --- Statements (English API) ---

class Statement(BaseModel):
    id: str
    politicianName: str
    politicianParty: str
    photoUrl: str | None = None
    statementText: str
    verdict: str  # "true" | "false" | "misleading" | "uncheckable"
    date: str     # ISO date YYYY-MM-DD
    sourceUrl: str | None = None
    explanation: str
    topic: str


# --- Chat ---

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


# --- Dashboard ---

class PartyStats(BaseModel):
    party: str
    true: int
    false: int
    misleading: int
    uncheckable: int
    total: int


class TopicStats(BaseModel):
    topic: str
    total: int
    true: int
    false: int
    misleading: int
    uncheckable: int


class PoliticianStats(BaseModel):
    id: str
    name: str
    party: str
    photoUrl: str | None = None
    total: int
    true: int
    false: int
    misleading: int
    uncheckable: int
    truthRate: float


class DashboardStats(BaseModel):
    totalStatements: int
    truthRate: float
    falseRate: float
    verdictBreakdown: dict[str, int]
    byParty: list[PartyStats]
    byTopic: list[TopicStats]
    byPolitician: list[PoliticianStats]
