import re

from contract_inspector.data.schemas import ChunkRecord
from contract_inspector.retrieval.bm25 import BM25ChunkRetriever
from contract_inspector.retrieval.schemas import RetrievalHit


class RuntimeDocumentTools:
    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self.chunks = chunks
        self.by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.by_contract: dict[str, list[ChunkRecord]] = {}
        for chunk in chunks:
            self.by_contract.setdefault(chunk.contract_id, []).append(chunk)
        self.retriever = BM25ChunkRetriever().fit(chunks)
        self.trace: list[dict] = []

    def search_chunks(self, query: str, contract_id: str, top_k: int = 5) -> list[RetrievalHit]:
        hits = self.retriever.search(query, contract_id=contract_id, top_k=top_k)
        self.trace.append({"tool": "search_chunks", "query": query, "hits": [hit.chunk_id for hit in hits]})
        return hits

    def read_chunk(self, chunk_id: str) -> ChunkRecord:
        self.trace.append({"tool": "read_chunk", "chunk_id": chunk_id})
        return self.by_id[chunk_id]

    def get_neighbors(self, chunk_id: str, left: int = 1, right: int = 1) -> list[ChunkRecord]:
        chunk = self.by_id[chunk_id]
        chunks = self.by_contract.get(chunk.contract_id, [])
        start = max(0, chunk.chunk_index - left)
        end = chunk.chunk_index + right + 1
        neighbors = chunks[start:end]
        self.trace.append({"tool": "get_neighbors", "chunk_id": chunk_id, "returned": [c.chunk_id for c in neighbors]})
        return neighbors

    def grep_contract(self, contract_id: str, pattern: str) -> list[dict]:
        regex = re.compile(pattern, flags=re.IGNORECASE)
        matches = []
        for chunk in self.by_contract.get(contract_id, []):
            for match in regex.finditer(chunk.text):
                matches.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "char_start": chunk.char_start + match.start(),
                        "char_end": chunk.char_start + match.end(),
                        "text": match.group(0),
                    }
                )
        self.trace.append({"tool": "grep_contract", "pattern": pattern, "matches": len(matches)})
        return matches
