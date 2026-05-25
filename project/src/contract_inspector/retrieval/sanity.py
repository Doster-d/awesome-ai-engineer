from contract_inspector.data.schemas import ChunkRecord
from contract_inspector.retrieval.schemas import RetrievalHit


class FirstChunksRetriever:
    def __init__(self) -> None:
        self.chunks: list[ChunkRecord] = []

    def fit(self, chunks: list[ChunkRecord]) -> "FirstChunksRetriever":
        self.chunks = chunks
        return self

    def search(
        self,
        query: str,
        contract_id: str | None = None,
        top_k: int = 10,
    ) -> list[RetrievalHit]:
        candidates = [chunk for chunk in self.chunks if contract_id is None or chunk.contract_id == contract_id]
        return [
            RetrievalHit(
                query=query,
                chunk_id=chunk.chunk_id,
                contract_id=chunk.contract_id,
                score=1.0 / rank,
                rank=rank,
                text=chunk.text,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                metadata={"retriever": "first_chunks"},
            )
            for rank, chunk in enumerate(candidates[:top_k], start=1)
        ]
