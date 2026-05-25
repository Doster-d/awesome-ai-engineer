import json
from collections import defaultdict

from pydantic import TypeAdapter

from contract_inspector.data.io import read_jsonl
from contract_inspector.data.normalize_ru_legal_qa import write_ru_legal_qa
from contract_inspector.data.schemas import ChunkRecord, GoldSpan
from contract_inspector.features.clause_queries import build_clause_query
from contract_inspector.retrieval.bm25 import BM25ChunkRetriever
from contract_inspector.retrieval.evaluate import (
    chunking_coverage,
    recall_at_k,
    reciprocal_rank,
    relevant_chunk_ids_for_example,
)
from contract_inspector.retrieval.sanity import FirstChunksRetriever
from contract_inspector.retrieval.tfidf import TfidfChunkRetriever
from contract_inspector.settings.paths import DATA_DIR, PROJECT_DIR


def evaluate_ru_legal_qa(top_k: int = 10) -> dict:
    write_ru_legal_qa()
    chunks = [ChunkRecord.model_validate(row) for row in read_jsonl(DATA_DIR / "processed" / "ru_legal_qa.chunks.jsonl")]
    examples = [
        row
        for row in read_jsonl(DATA_DIR / "processed" / "ru_legal_qa.clause_examples.jsonl")
        if row.get("gold_spans")
    ]
    chunks_by_contract: dict[str, list[ChunkRecord]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_contract[chunk.contract_id].append(chunk)

    retrievers = {
        "sanity_first_chunks": FirstChunksRetriever().fit(chunks),
        "tfidf": TfidfChunkRetriever().fit(chunks),
        "bm25": BM25ChunkRetriever().fit(chunks),
    }
    metrics = {
        "dataset": "ru_legal_qa_v1",
        "coverage": chunking_coverage(examples, chunks_by_contract),
        "retrievers": {},
        "examples_evaluated": len(examples),
    }
    for name, retriever in retrievers.items():
        totals = {"recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0}
        evaluated = 0
        for example in examples:
            spans = TypeAdapter(list[GoldSpan]).validate_python(example["gold_spans"])
            relevant_ids = relevant_chunk_ids_for_example(chunks_by_contract[example["contract_id"]], spans)
            if not relevant_ids:
                continue
            query = example.get("metadata", {}).get("question") or build_clause_query(example["clause_type"])
            hits = retriever.search(query, contract_id=example["contract_id"], top_k=top_k)
            totals["recall@1"] += recall_at_k(hits, relevant_ids, 1)
            totals["recall@3"] += recall_at_k(hits, relevant_ids, 3)
            totals["recall@5"] += recall_at_k(hits, relevant_ids, 5)
            totals["recall@10"] += recall_at_k(hits, relevant_ids, 10)
            totals["mrr"] += reciprocal_rank(hits, relevant_ids)
            evaluated += 1
        metrics["retrievers"][name] = {
            key: value / evaluated if evaluated else 0.0 for key, value in totals.items()
        } | {"evaluated": evaluated}
    return metrics


def main() -> None:
    metrics = evaluate_ru_legal_qa()
    output_path = PROJECT_DIR / "artifacts" / "ru_legal_qa_retrieval_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
