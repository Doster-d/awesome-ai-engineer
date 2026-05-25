import math
import re
from collections import Counter, defaultdict

from contract_inspector.data.schemas import ChunkRecord
from contract_inspector.retrieval.schemas import RetrievalHit


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25ChunkRetriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[ChunkRecord] = []
        self.doc_terms: list[Counter[str]] = []
        self.doc_lens: list[int] = []
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0

    def fit(self, chunks: list[ChunkRecord]) -> "BM25ChunkRetriever":
        self.chunks = chunks
        self.doc_terms = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self.doc_lens = [sum(terms.values()) for terms in self.doc_terms]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0

        df: defaultdict[str, int] = defaultdict(int)
        for terms in self.doc_terms:
            for term in terms:
                df[term] += 1

        doc_count = len(chunks)
        self.idf = {
            term: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        return self

    def _score_doc(self, query_terms: list[str], doc_index: int) -> float:
        if not self.avgdl:
            return 0.0

        terms = self.doc_terms[doc_index]
        doc_len = self.doc_lens[doc_index]
        score = 0.0
        for term in query_terms:
            tf = terms.get(term, 0)
            if tf == 0:
                continue
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / denom
        return score

    def search(
        self,
        query: str,
        contract_id: str | None = None,
        top_k: int = 10,
    ) -> list[RetrievalHit]:
        if not self.chunks or top_k <= 0:
            return []

        query_terms = tokenize(query)
        candidate_indexes = [
            idx for idx, chunk in enumerate(self.chunks) if contract_id is None or chunk.contract_id == contract_id
        ]
        scored = [(idx, self._score_doc(query_terms, idx)) for idx in candidate_indexes]
        ranked = sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]

        return [
            RetrievalHit(
                query=query,
                chunk_id=self.chunks[idx].chunk_id,
                contract_id=self.chunks[idx].contract_id,
                score=float(score),
                rank=rank,
                text=self.chunks[idx].text,
                char_start=self.chunks[idx].char_start,
                char_end=self.chunks[idx].char_end,
                metadata={"retriever": "bm25"},
            )
            for rank, (idx, score) in enumerate(ranked, start=1)
        ]
