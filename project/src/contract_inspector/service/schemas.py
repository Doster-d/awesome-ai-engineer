from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    contract_text: str
    clause_types: list[str] = Field(default_factory=list)
    top_k: int = 5
    use_rlm: bool = False


class EvidenceItem(BaseModel):
    chunk_id: str
    char_start: int
    char_end: int
    quote: str
    score: float
    source: str


class ClauseResult(BaseModel):
    clause_type: str
    found: bool
    answer: str
    evidence: list[EvidenceItem]
    confidence: float
    risk_level: str | None = None
    rlm_trace: list[dict] = Field(default_factory=list)


class PredictResponse(BaseModel):
    contract_id: str
    results: list[ClauseResult]
    model_version: str
