import argparse
import json
import os
from collections import defaultdict

import dspy
from pydantic import TypeAdapter

from contract_inspector.data.io import read_jsonl
from contract_inspector.data.schemas import ChunkRecord, GoldSpan
from contract_inspector.features.clause_queries import build_clause_query
from contract_inspector.retrieval.bm25 import BM25ChunkRetriever
from contract_inspector.retrieval.evaluate import relevant_chunk_ids_for_example
from contract_inspector.rlm.config import configure_dspy_from_env
from contract_inspector.rlm.programs import ClauseEvidenceProgram
from contract_inspector.settings.domain_config import load_rlm_feedback
from contract_inspector.settings.paths import DATA_DIR, PROJECT_DIR


def build_gepa_trainset(dataset: str = "ru_legal_qa", limit: int = 80, top_k: int = 10, hard_only: bool = True) -> list[dspy.Example]:
    chunks_path = DATA_DIR / "processed" / f"{dataset}.chunks.jsonl"
    examples_path = DATA_DIR / "processed" / f"{dataset}.clause_examples.jsonl"
    if not chunks_path.exists() or not examples_path.exists():
        chunks_path = DATA_DIR / "processed" / "chunks.jsonl"
        examples_path = DATA_DIR / "processed" / "clause_examples.jsonl"

    chunks = [ChunkRecord.model_validate(row) for row in read_jsonl(chunks_path)]
    examples = [row for row in read_jsonl(examples_path) if row.get("gold_spans")]
    chunks_by_contract: dict[str, list[ChunkRecord]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_contract[chunk.contract_id].append(chunk)
    retriever = BM25ChunkRetriever().fit(chunks)

    trainset = []
    for example in examples:
        spans = TypeAdapter(list[GoldSpan]).validate_python(example["gold_spans"])
        query = example.get("metadata", {}).get("question") or build_clause_query(example["clause_type"])
        hits = retriever.search(query, contract_id=example["contract_id"], top_k=top_k)
        if hard_only:
            relevant_ids = relevant_chunk_ids_for_example(chunks_by_contract[example["contract_id"]], spans)
            if hits[:3] and any(hit.chunk_id in relevant_ids for hit in hits[:3]):
                continue
            if not any(hit.chunk_id in relevant_ids for hit in hits):
                continue
        evidence_items = [
            {
                "chunk_id": hit.chunk_id,
                "quote": _clip_text(hit.text, 1400),
                "context": _clip_text(_neighbor_context(chunks_by_contract[hit.contract_id], hit.chunk_id), 2200),
                "char_start": hit.char_start,
                "char_end": hit.char_end,
                "score": hit.score,
            }
            for hit in hits
        ]
        trainset.append(
            dspy.Example(
                clause_type=query,
                evidence_quotes=json.dumps(evidence_items, ensure_ascii=False),
                gold_quotes=json.dumps([span.text for span in spans], ensure_ascii=False),
            ).with_inputs("clause_type", "evidence_quotes")
        )
        if len(trainset) >= limit:
            break
    return trainset


def evidence_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    gold_quotes = json.loads(gold.gold_quotes)
    answer_json = str(getattr(pred, "answer_json", ""))
    judge_outputs = str(getattr(pred, "judge_outputs", ""))
    answer_text = f"{answer_json} {judge_outputs}".lower()
    quote_score = max((_quote_match_score(quote, answer_text) for quote in gold_quotes), default=0.0)
    found_score = 1.0 if '"found": true' in answer_json.lower() or '"found":true' in answer_json.lower() else 0.0
    score = 0.7 * quote_score + 0.3 * found_score * quote_score
    feedback = load_rlm_feedback()["gepa"]["feedback_template"].format(quote_score=quote_score, score=score)
    return dspy.Prediction(score=score, feedback=feedback)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize the DSPy RLM program with GEPA.")
    parser.add_argument("--dataset", default=os.getenv("GEPA_DATASET", "ru_legal_qa"))
    parser.add_argument("--limit", type=int, default=int(os.getenv("GEPA_TRAIN_LIMIT", "40")))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("GEPA_TOP_K", "10")))
    parser.add_argument("--auto", default=os.getenv("GEPA_AUTO", "light"), choices=["light", "medium", "heavy"])
    parser.add_argument("--max-metric-calls", type=int, default=None)
    parser.add_argument("--all-examples", action="store_true")
    args = parser.parse_args()

    student_lm = configure_dspy_from_env(role="student", temperature=0.0, max_tokens=3000)
    teacher_lm = configure_dspy_from_env(role="teacher", temperature=0.2, max_tokens=2000)
    if student_lm is None or teacher_lm is None:
        raise RuntimeError("Set LITELLM_API_KEY and OPENROUTER_API_KEY before running GEPA optimization.")
    dspy.configure(lm=student_lm)

    trainset = build_gepa_trainset(dataset=args.dataset, limit=args.limit, top_k=args.top_k, hard_only=not args.all_examples)
    optimizer = dspy.GEPA(
        metric=evidence_metric,
        auto=None if args.max_metric_calls is not None else args.auto,
        max_metric_calls=args.max_metric_calls,
        reflection_lm=teacher_lm,
        log_dir=str(PROJECT_DIR / "artifacts" / "gepa_logs"),
        track_stats=True,
    )
    compiled = optimizer.compile(ClauseEvidenceProgram(), trainset=trainset)
    output_path = PROJECT_DIR / "artifacts" / "rlm_gepa_program.json"
    compiled.save(str(output_path))
    (PROJECT_DIR / "artifacts" / "rlm_gepa_summary.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "train_examples": len(trainset),
                "top_k": args.top_k,
                "hard_only": not args.all_examples,
                "output": str(output_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved GEPA-optimized RLM program to {output_path}")


def _neighbor_context(chunks: list[ChunkRecord], chunk_id: str) -> str:
    index = next((idx for idx, chunk in enumerate(chunks) if chunk.chunk_id == chunk_id), 0)
    return "\n\n".join(chunk.text for chunk in chunks[max(0, index - 1) : index + 2])


def _quote_match_score(quote: str, answer_text: str) -> float:
    quote = " ".join(quote.lower().split())
    if not quote:
        return 0.0
    if quote[:120] in answer_text or quote[-120:] in answer_text:
        return 1.0
    terms = {term for term in quote.split() if len(term) > 6}
    if not terms:
        return 0.0
    matched = sum(1 for term in terms if term in answer_text)
    return matched / len(terms)


def _clip_text(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ..."


if __name__ == "__main__":
    main()
