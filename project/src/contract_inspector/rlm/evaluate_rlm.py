import argparse
import json
from pathlib import Path

from contract_inspector.data.io import read_jsonl
from contract_inspector.data.normalize_ru_legal_qa import load_processed_ru_legal_qa
from contract_inspector.data.schemas import ClauseExample
from contract_inspector.service.pipeline import ContractInspectionPipeline
from contract_inspector.settings.paths import PROJECT_DIR


def evaluate_rlm_ru_legal_qa(
    limit_examples: int = 20,
    top_k: int = 3,
    rlm_top_k: int | None = None,
    hard_only: bool = False,
    examples_path: Path | None = None,
) -> dict:
    contracts, examples = load_processed_ru_legal_qa()
    contracts_by_id = {contract.contract_id: contract for contract in contracts}
    if examples_path is not None:
        examples = [ClauseExample.model_validate(row) for row in read_jsonl(examples_path)]
    examples = [example for example in examples if example.gold_spans]
    pipeline = ContractInspectionPipeline()

    baseline_rows = []
    for example in examples:
        contract = contracts_by_id[example.contract_id]
        query = str(example.metadata.get("question") or example.clause_type)
        baseline = pipeline.inspect(contract.text, [query], top_k=top_k, use_rlm=False)
        baseline_item = baseline["results"][0]
        baseline_rows.append(
            {
                "example": example,
                "query": query,
                "baseline_found": bool(baseline_item["found"]),
                "baseline_supported": _evidence_supports_gold(baseline_item["evidence"], example),
            }
        )

    if hard_only:
        selected_rows = [row for row in baseline_rows if not row["baseline_supported"]]
    else:
        selected_rows = baseline_rows
    selected_rows = selected_rows[:limit_examples]

    rows = []
    for index, selected in enumerate(selected_rows, start=1):
        example = selected["example"]
        contract = contracts_by_id[example.contract_id]
        query = selected["query"]
        print(f"RLM example {index}/{len(selected_rows)}: {example.example_id}", flush=True)
        rlm = pipeline.inspect(contract.text, [query], top_k=rlm_top_k or top_k, use_rlm=True)
        rlm_item = rlm["results"][0]
        rows.append(
            {
                "example_id": example.example_id,
                "contract_id": example.contract_id,
                "question": query,
                "baseline_found": selected["baseline_found"],
                "rlm_found": bool(rlm_item["found"]),
                "baseline_supported": selected["baseline_supported"],
                "rlm_supported": _evidence_supports_gold(rlm_item["evidence"], example),
                "rlm_enabled": bool(rlm_item.get("rlm_trace")),
            }
        )

    return {
        "dataset": "ru_legal_qa_v1",
        "task": "question_to_legal_quote_support",
        "selection": "baseline_errors" if hard_only else "first_examples",
        "examples_path": str(examples_path) if examples_path is not None else None,
        "baseline_top_k": top_k,
        "rlm_top_k": rlm_top_k or top_k,
        "baseline_errors_total": sum(not row["baseline_supported"] for row in baseline_rows),
        "baseline_examples_total": len(baseline_rows),
        "rows": rows,
        "baseline": _summarize(rows, "baseline"),
        "rlm": _summarize(rows, "rlm"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline versus RLM on Ru-Legal-QA-v1.")
    parser.add_argument("--limit-examples", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rlm-top-k", type=int, default=None)
    parser.add_argument("--hard-only", action="store_true", help="Run RLM only on examples where the baseline missed gold quotes.")
    parser.add_argument("--examples-path", type=Path, default=None)
    args = parser.parse_args()

    metrics = evaluate_rlm_ru_legal_qa(
        limit_examples=args.limit_examples,
        top_k=args.top_k,
        rlm_top_k=args.rlm_top_k,
        hard_only=args.hard_only,
        examples_path=args.examples_path,
    )
    output_path = PROJECT_DIR / "artifacts" / "ru_legal_qa_rlm_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in metrics.items() if key != "rows"}, indent=2, ensure_ascii=False))


def _summarize(rows: list[dict], prefix: str) -> dict:
    return {
        "found_rate": _mean(row[f"{prefix}_found"] for row in rows),
        "quote_support_rate": _mean(row[f"{prefix}_supported"] for row in rows),
        "answer_support_rate": _mean(row[f"{prefix}_found"] and row[f"{prefix}_supported"] for row in rows),
        "evaluated": len(rows),
        "rlm_enabled_rate": _mean(row["rlm_enabled"] for row in rows) if prefix == "rlm" else None,
    }


def _mean(values) -> float:
    values = list(values)
    return sum(bool(value) for value in values) / len(values) if values else 0.0


def _evidence_supports_gold(evidence: list[dict], example: ClauseExample) -> bool:
    for item in evidence:
        for span in example.gold_spans:
            if item["char_start"] < span.end and item["char_end"] > span.start:
                return True
    return False


if __name__ == "__main__":
    main()
