import argparse
import json
from pathlib import Path

from contract_inspector.data.io import write_jsonl
from contract_inspector.data.normalize_ru_legal_qa import load_processed_ru_legal_qa
from contract_inspector.data.schemas import ClauseExample
from contract_inspector.service.pipeline import ContractInspectionPipeline
from contract_inspector.settings.paths import DATA_DIR, PROJECT_DIR


def build_rlm_validation_set(
    limit: int | None = None,
    baseline_top_k: int = 3,
    candidate_top_k: int = 10,
    exclude_first: int = 0,
    target_baseline_support_rate: float = 0.125,
) -> dict:
    contracts, examples = load_processed_ru_legal_qa()
    contracts_by_id = {contract.contract_id: contract for contract in contracts}
    examples = [example for example in examples if example.gold_spans]
    pipeline = ContractInspectionPipeline()

    supported_pool = []
    missed_pool = []
    for example in examples:
        contract = contracts_by_id[example.contract_id]
        query = str(example.metadata.get("question") or example.clause_type)
        baseline = pipeline.inspect(contract.text, [query], top_k=baseline_top_k, use_rlm=False)
        candidate = pipeline.inspect(contract.text, [query], top_k=candidate_top_k, use_rlm=False)
        baseline_item = baseline["results"][0]
        candidate_item = candidate["results"][0]
        baseline_supported = _evidence_supports_gold(baseline_item["evidence"], example)
        candidate_supported = _evidence_supports_gold(candidate_item["evidence"], example)
        if not candidate_supported:
            continue
        row = _validation_row(
            example=example,
            baseline_item=baseline_item,
            candidate_item=candidate_item,
            baseline_supported=baseline_supported,
            candidate_supported=candidate_supported,
            baseline_top_k=baseline_top_k,
            candidate_top_k=candidate_top_k,
        )
        if baseline_supported:
            supported_pool.append(row)
        else:
            missed_pool.append(row)

    missed_pool = missed_pool[exclude_first:]
    if limit is None:
        supported_target = 0
        missed_target = len(missed_pool)
    else:
        supported_target = round(limit * target_baseline_support_rate)
        supported_target = min(supported_target, len(supported_pool))
        missed_target = max(0, limit - supported_target)
    selected = missed_pool[:missed_target] + supported_pool[:supported_target]

    output_path = DATA_DIR / "processed" / "rlm_validation_examples.jsonl"
    artifact_path = PROJECT_DIR / "artifacts" / "rlm_validation_summary.json"
    write_jsonl(output_path, selected)
    summary = {
        "dataset": "ru_legal_qa_v1",
        "selection_rule": "mixed_baseline_success_and_hard_cases_without_rlm_selection",
        "baseline_top_k": baseline_top_k,
        "candidate_top_k": candidate_top_k,
        "exclude_first": exclude_first,
        "target_baseline_support_rate": target_baseline_support_rate,
        "examples": len(selected),
        "baseline_supported_examples": sum(row["selection_metrics"]["baseline_supported"] for row in selected),
        "baseline_answer_support_rate_by_selection": _mean(
            row["selection_metrics"]["baseline_supported"] for row in selected
        ),
        "candidate_answer_support_rate_by_selection": 1.0 if selected else 0.0,
        "uses_rlm_for_selection": False,
        "output": str(output_path),
    }
    artifact_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a non-leaky RLM validation set from Ru-Legal-QA hard cases.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--baseline-top-k", type=int, default=3)
    parser.add_argument("--candidate-top-k", type=int, default=10)
    parser.add_argument("--exclude-first", type=int, default=0)
    parser.add_argument("--target-baseline-support-rate", type=float, default=0.125)
    args = parser.parse_args()
    print(
        json.dumps(
            build_rlm_validation_set(
                limit=args.limit,
                baseline_top_k=args.baseline_top_k,
                candidate_top_k=args.candidate_top_k,
                exclude_first=args.exclude_first,
                target_baseline_support_rate=args.target_baseline_support_rate,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


def _evidence_supports_gold(evidence: list[dict], example: ClauseExample) -> bool:
    for item in evidence:
        for span in example.gold_spans:
            if item["char_start"] < span.end and item["char_end"] > span.start:
                return True
    return False


def _validation_row(
    example: ClauseExample,
    baseline_item: dict,
    candidate_item: dict,
    baseline_supported: bool,
    candidate_supported: bool,
    baseline_top_k: int,
    candidate_top_k: int,
) -> dict:
    return {
        "example_id": example.example_id,
        "contract_id": example.contract_id,
        "clause_type": example.clause_type,
        "label_present": example.label_present,
        "answer_value": example.answer_value,
        "gold_spans": [span.model_dump(mode="json") for span in example.gold_spans],
        "metadata": {
            **example.metadata,
            "selection_rule": "mixed_baseline_success_and_hard_cases_without_rlm_selection",
            "baseline_top_k": baseline_top_k,
            "candidate_top_k": candidate_top_k,
        },
        "selection_metrics": {
            "baseline_found": bool(baseline_item["found"]),
            "baseline_supported": baseline_supported,
            "candidate_found": bool(candidate_item["found"]),
            "candidate_supported": candidate_supported,
        },
    }


def _mean(values) -> float:
    values = list(values)
    return sum(bool(value) for value in values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
