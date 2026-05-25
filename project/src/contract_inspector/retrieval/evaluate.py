import json
from collections import defaultdict

from pydantic import TypeAdapter

from contract_inspector.data.io import read_jsonl
from contract_inspector.data.schemas import ChunkRecord, GoldSpan
from contract_inspector.features.clause_queries import build_clause_query
from contract_inspector.retrieval.bm25 import BM25ChunkRetriever
from contract_inspector.retrieval.schemas import RetrievalHit
from contract_inspector.retrieval.sanity import FirstChunksRetriever
from contract_inspector.retrieval.tfidf import TfidfChunkRetriever
from contract_inspector.settings.paths import DATA_DIR, PROJECT_DIR


def span_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def span_overlap_ratio(gold_start: int, gold_end: int, chunk_start: int, chunk_end: int) -> float:
    overlap = span_overlap(gold_start, gold_end, chunk_start, chunk_end)
    gold_len = gold_end - gold_start
    if gold_len <= 0:
        return 0.0
    return overlap / gold_len


def is_relevant_chunk(
    chunk: ChunkRecord,
    gold_spans: list[GoldSpan],
    min_overlap_ratio: float = 0.1,
) -> bool:
    return any(
        span_overlap_ratio(span.start, span.end, chunk.char_start, chunk.char_end)
        >= min_overlap_ratio
        for span in gold_spans
    )


def recall_at_k(hits: list[RetrievalHit], relevant_chunk_ids: set[str], k: int) -> float:
    if not relevant_chunk_ids:
        return 0.0
    returned = {hit.chunk_id for hit in hits[:k]}
    return float(bool(returned & relevant_chunk_ids))


def reciprocal_rank(hits: list[RetrievalHit], relevant_chunk_ids: set[str]) -> float:
    for hit in hits:
        if hit.chunk_id in relevant_chunk_ids:
            return 1.0 / hit.rank
    return 0.0


def relevant_chunk_ids_for_example(
    chunks: list[ChunkRecord],
    gold_spans: list[GoldSpan],
    min_overlap_ratio: float = 0.1,
) -> set[str]:
    return {chunk.chunk_id for chunk in chunks if is_relevant_chunk(chunk, gold_spans, min_overlap_ratio)}


def chunking_coverage(
    examples: list[dict],
    chunks_by_contract: dict[str, list[ChunkRecord]],
) -> dict:
    total = 0
    covered = 0
    max_ratios = []
    for example in examples:
        spans = TypeAdapter(list[GoldSpan]).validate_python(example.get("gold_spans", []))
        if not spans:
            continue
        for span in spans:
            total += 1
            ratios = [
                span_overlap_ratio(span.start, span.end, chunk.char_start, chunk.char_end)
                for chunk in chunks_by_contract.get(example["contract_id"], [])
            ]
            max_ratio = max(ratios) if ratios else 0.0
            max_ratios.append(max_ratio)
            covered += int(max_ratio >= 0.999)

    return {
        "gold_spans": total,
        "gold_span_covered": covered,
        "coverage_rate": covered / total if total else 0.0,
        "max_overlap_ratio_mean": sum(max_ratios) / len(max_ratios) if max_ratios else 0.0,
        "evidence_loss_rate": 1 - covered / total if total else 0.0,
    }


def evaluate_retrievers(top_k: int = 10, max_examples: int | None = None) -> dict:
    chunks = [ChunkRecord.model_validate(row) for row in read_jsonl(DATA_DIR / "processed" / "chunks.jsonl")]
    examples = list(read_jsonl(DATA_DIR / "processed" / "clause_examples.jsonl"))
    examples = [example for example in examples if example.get("gold_spans")]
    if max_examples:
        examples = examples[:max_examples]

    chunks_by_contract: dict[str, list[ChunkRecord]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_contract[chunk.contract_id].append(chunk)

    retrievers = {
        "sanity_first_chunks": FirstChunksRetriever().fit(chunks),
        "tfidf": TfidfChunkRetriever().fit(chunks),
        "bm25": BM25ChunkRetriever().fit(chunks),
    }

    metrics = {
        "coverage": chunking_coverage(examples, chunks_by_contract),
        "retrievers": {},
        "examples_evaluated": len(examples),
    }
    for name, retriever in retrievers.items():
        totals = {"recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0}
        evaluated = 0
        for example in examples:
            contract_chunks = chunks_by_contract[example["contract_id"]]
            spans = TypeAdapter(list[GoldSpan]).validate_python(example["gold_spans"])
            relevant_ids = relevant_chunk_ids_for_example(contract_chunks, spans)
            if not relevant_ids:
                continue
            query = build_clause_query(example["clause_type"])
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
    artifacts_dir = PROJECT_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_retrievers()
    (artifacts_dir / "chunking_coverage.json").write_text(
        json.dumps(metrics["coverage"], indent=2),
        encoding="utf-8",
    )
    (artifacts_dir / "retrieval_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
