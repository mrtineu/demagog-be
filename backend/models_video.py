from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional


class VerificationMode(str, Enum):
    db_only = "db_only"
    full = "full"


class JobStatus(str, Enum):
    pending = "pending"
    downloading_audio = "downloading_audio"
    extracting_audio = "extracting_audio"
    transcribing = "transcribing"
    extracting_statements = "extracting_statements"
    verifying = "verifying"
    completed = "completed"
    failed = "failed"


class Verdict(str, Enum):
    pravda = "Pravda"
    nepravda = "Nepravda"
    zavadzajuce = "Zavádzajúce"
    neoveritelne = "Neoveriteľné"
    nedostatok_dat = "Nedostatok dát"


# --- Transcript models ---


class TranscriptSegment(BaseModel):
    start_time: float  # seconds
    end_time: float  # seconds
    text: str


class Transcript(BaseModel):
    segments: list[TranscriptSegment]
    full_text: str
    language: str = "sk"
    duration_seconds: float


# --- Statement extraction models ---


class ExtractedStatement(BaseModel):
    text: str
    speaker: Optional[str] = None
    start_time: float
    end_time: float
    segment_indices: list[int] = Field(
        default_factory=list,
        description="Indices into the transcript segments this statement spans",
    )


# --- Verification result models ---


class SourceInfo(BaseModel):
    vyrok: Optional[str] = None
    vyhodnotenie: Optional[str] = None
    odovodnenie: Optional[str] = None
    meno: Optional[str] = None
    politicka_strana: Optional[str] = None
    datum: Optional[str] = None
    skore_podobnosti: Optional[float] = None


class WebSource(BaseModel):
    url: str
    nazov: Optional[str] = None
    relevantny_citat: Optional[str] = None
    typ_zdroja: Optional[str] = None


class VerifiedStatement(BaseModel):
    statement_text: str
    speaker: Optional[str] = None
    start_time: float
    end_time: float
    verdict: str
    verdict_label: str
    reasoning: str
    verification_type: str  # "databaza" or "webovy_vyskum"
    confidence: Optional[str] = None  # for web research results
    db_source: Optional[SourceInfo] = None
    web_sources: list[WebSource] = Field(default_factory=list)


# --- Job models ---


class JobProgress(BaseModel):
    job_id: str
    status: JobStatus
    progress_percent: int = Field(default=0, ge=0, le=100)
    current_step: str = ""
    statements_total: Optional[int] = None
    statements_verified: Optional[int] = None
    error_message: Optional[str] = None


class VideoAnalysisResponse(BaseModel):
    job_id: str
    status: JobStatus
    transcript: Optional[Transcript] = None
    extracted_statements: list[ExtractedStatement] = Field(default_factory=list)
    verified_statements: list[VerifiedStatement] = Field(default_factory=list)
    video_duration_seconds: Optional[float] = None
    processing_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
