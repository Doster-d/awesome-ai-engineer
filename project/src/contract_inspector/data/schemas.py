from pydantic import BaseModel, Field


class ContractDoc(BaseModel):
    contract_id: str = Field(..., description="Stable contract id")
    dataset: str = Field(..., description="Dataset name: cuad, contractnli, runtime")
    split: str = Field(..., description="train/dev/test/runtime")
    source_file: str | None = None
    name: str | None = None
    text: str
    metadata: dict = Field(default_factory=dict)


class GoldSpan(BaseModel):
    start: int
    end: int
    text: str


class ClauseExample(BaseModel):
    example_id: str
    contract_id: str
    clause_type: str
    label_present: bool
    answer_value: str | None = None
    gold_spans: list[GoldSpan] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class NLIExample(BaseModel):
    example_id: str
    contract_id: str
    hypothesis_id: str
    hypothesis: str
    gold_label: str
    gold_spans: list[GoldSpan] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class ChunkRecord(BaseModel):
    chunk_id: str
    contract_id: str
    chunk_index: int
    char_start: int
    char_end: int
    text: str
    word_count: int
    metadata: dict = Field(default_factory=dict)

