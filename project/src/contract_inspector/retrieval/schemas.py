from pydantic import BaseModel, Field


class RetrievalHit(BaseModel):
    query: str
    chunk_id: str
    contract_id: str
    score: float
    rank: int
    text: str
    char_start: int
    char_end: int
    metadata: dict = Field(default_factory=dict)
