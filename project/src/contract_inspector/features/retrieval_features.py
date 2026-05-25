import re
from collections import defaultdict

import pandas as pd
from pydantic import TypeAdapter

from contract_inspector.data.io import read_jsonl
from contract_inspector.data.schemas import ChunkRecord
from contract_inspector.data.schemas import GoldSpan
from contract_inspector.features.clause_queries import build_clause_query
from contract_inspector.retrieval.bm25 import BM25ChunkRetriever
from contract_inspector.retrieval.evaluate import is_relevant_chunk
from contract_inspector.retrieval.schemas import RetrievalHit
from contract_inspector.retrieval.tfidf import TfidfChunkRetriever
from contract_inspector.settings.paths import DATA_DIR, PROJECT_DIR


LEGAL_MARKERS = {"shall", "hereby", "agreement", "liability", "law", "consent", "terminate"}


def query_terms(query: str) -> set[str]:
    return set(re.findall(r"\w+", query.lower()))


def build_retrieval_feature_row(
    query: str,
    clause_type: str,
    chunk: ChunkRecord,
    hit: RetrievalHit | None = None,
    is_relevant: bool | None = None,
) -> dict:
    terms = query_terms(query)
    chunk_terms = query_terms(chunk.text)
    matched = terms & chunk_terms
    return {
        "query": query,
        "clause_type": clause_type,
        "contract_id": chunk.contract_id,
        "chunk_id": chunk.chunk_id,
        "score": hit.score if hit else 0.0,
        "rank": hit.rank if hit else 0,
        "query_term_coverage": len(matched) / len(terms) if terms else 0.0,
        "matched_query_terms_count": len(matched),
        "legal_marker_count": sum(1 for term in chunk_terms if term in LEGAL_MARKERS),
        "word_count": chunk.word_count,
        "relative_position": chunk.chunk_index,
        "chunk_index": chunk.chunk_index,
        "number_count": len(re.findall(r"\b\d+(?:\.\d+)?\b", chunk.text)),
        "date_count": len(re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", chunk.text)),
        "money_count": len(re.findall(r"[$€£]\s?\d+", chunk.text)),
        "is_relevant": is_relevant,
    }


def build_baseline_candidates(top_k: int = 10) -> pd.DataFrame:
    chunks = [ChunkRecord.model_validate(row) for row in read_jsonl(DATA_DIR / "processed" / "chunks.jsonl")]
    examples = [row for row in read_jsonl(DATA_DIR / "processed" / "clause_examples.jsonl") if row.get("gold_spans")]
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    chunks_by_contract: dict[str, list[ChunkRecord]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_contract[chunk.contract_id].append(chunk)

    tfidf = TfidfChunkRetriever().fit(chunks)
    bm25 = BM25ChunkRetriever().fit(chunks)
    rows = []
    for example in examples:
        query = build_clause_query(example["clause_type"])
        spans = TypeAdapter(list[GoldSpan]).validate_python(example["gold_spans"])
        tfidf_hits = tfidf.search(query, contract_id=example["contract_id"], top_k=top_k)
        bm25_hits = bm25.search(query, contract_id=example["contract_id"], top_k=top_k)
        by_chunk: dict[str, dict] = {}
        for hit in tfidf_hits:
            by_chunk.setdefault(hit.chunk_id, {})["tfidf_hit"] = hit
        for hit in bm25_hits:
            by_chunk.setdefault(hit.chunk_id, {})["bm25_hit"] = hit

        for chunk_id, hit_data in by_chunk.items():
            chunk = chunks_by_id[chunk_id]
            tfidf_hit = hit_data.get("tfidf_hit")
            bm25_hit = hit_data.get("bm25_hit")
            base_hit = bm25_hit or tfidf_hit
            row = build_retrieval_feature_row(
                query=query,
                clause_type=example["clause_type"],
                chunk=chunk,
                hit=base_hit,
                is_relevant=is_relevant_chunk(chunk, spans),
            )
            row.update(
                {
                    "example_id": example["example_id"],
                    "tfidf_score": tfidf_hit.score if tfidf_hit else 0.0,
                    "bm25_score": bm25_hit.score if bm25_hit else 0.0,
                    "tfidf_rank": tfidf_hit.rank if tfidf_hit else top_k + 1,
                    "bm25_rank": bm25_hit.rank if bm25_hit else top_k + 1,
                    "contract_chunk_count": len(chunks_by_contract[example["contract_id"]]),
                    "relative_position": chunk.chunk_index
                    / max(1, len(chunks_by_contract[example["contract_id"]]) - 1),
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    artifacts_dir = PROJECT_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    frame = build_baseline_candidates()
    frame.to_parquet(artifacts_dir / "baseline_candidates.parquet", index=False)
    print({"rows": len(frame), "positive_rate": float(frame["is_relevant"].mean()) if len(frame) else 0.0})


if __name__ == "__main__":
    main()
