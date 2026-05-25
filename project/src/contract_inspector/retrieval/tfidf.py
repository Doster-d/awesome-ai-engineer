import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from contract_inspector.data.schemas import ChunkRecord
from contract_inspector.retrieval.schemas import RetrievalHit


class TfidfChunkRetriever:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
        self.chunks: list[ChunkRecord] = []
        self.matrix = None

    def fit(self, chunks: list[ChunkRecord]) -> "TfidfChunkRetriever":
        self.chunks = chunks
        if chunks:
            self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in chunks])
        return self

    def search(
        self,
        query: str,
        contract_id: str | None = None,
        top_k: int = 10,
    ) -> list[RetrievalHit]:
        if self.matrix is None or not self.chunks or top_k <= 0:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vec.T).toarray().ravel()
        candidate_indexes = [
            idx for idx, chunk in enumerate(self.chunks) if contract_id is None or chunk.contract_id == contract_id
        ]
        ranked = sorted(candidate_indexes, key=lambda idx: scores[idx], reverse=True)[:top_k]

        return [
            RetrievalHit(
                query=query,
                chunk_id=self.chunks[idx].chunk_id,
                contract_id=self.chunks[idx].contract_id,
                score=float(np.asarray(scores[idx]).item()),
                rank=rank,
                text=self.chunks[idx].text,
                char_start=self.chunks[idx].char_start,
                char_end=self.chunks[idx].char_end,
                metadata={"retriever": "tfidf"},
            )
            for rank, idx in enumerate(ranked, start=1)
        ]
