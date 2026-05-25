from typing import Protocol

from contract_inspector.data.schemas import ChunkRecord
from contract_inspector.retrieval.schemas import RetrievalHit


class ChunkRetriever(Protocol):
    def fit(self, chunks: list[ChunkRecord]) -> "ChunkRetriever": ...

    def search(
        self,
        query: str,
        contract_id: str | None = None,
        top_k: int = 10,
    ) -> list[RetrievalHit]: ...
