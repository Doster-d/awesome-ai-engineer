from pydantic import BaseModel


class RuntimeConfig(BaseModel):
    chunk_size_words: int = 220
    overlap_words: int = 50
    top_k: int = 5
    model_version: str = "bm25_runtime_v1"
