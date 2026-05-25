import json

from contract_inspector.data.io import write_jsonl
from contract_inspector.data.normalize_contractnli import load_contractnli_zip
from contract_inspector.data.normalize_cuad import MVP_CLAUSE_TYPES, load_cuad_zip
from contract_inspector.data.normalize_ru_legal_qa import load_ru_legal_qa
from contract_inspector.data.schemas import ClauseExample, ContractDoc, GoldSpan
from contract_inspector.features.chunking import chunk_contract
from contract_inspector.settings.paths import DATA_DIR, PROJECT_DIR


def demo_contracts() -> tuple[list[ContractDoc], list[ClauseExample]]:
    text = (
        "This Agreement shall be governed by the laws of New York. "
        "Neither party may assign this Agreement without prior written consent. "
        "The aggregate liability of either party is capped at fees paid in the prior twelve months."
    )
    contract = ContractDoc(
        contract_id="demo_0001",
        dataset="runtime",
        split="train",
        source_file="demo",
        name="Demo Agreement",
        text=text,
    )
    clauses = []
    for clause_type, evidence in [
        ("Governing Law", "governed by the laws of New York"),
        ("Anti-Assignment", "may assign this Agreement without prior written consent"),
        ("Cap On Liability", "aggregate liability of either party is capped"),
    ]:
        start = text.find(evidence)
        clauses.append(
            ClauseExample(
                example_id=f"demo_0001__{clause_type.lower().replace(' ', '_').replace('-', '_')}",
                contract_id="demo_0001",
                clause_type=clause_type,
                label_present=True,
                answer_value="Yes",
                gold_spans=[GoldSpan(start=start, end=start + len(evidence), text=evidence)],
                metadata={"match_status": "exact", "source": "demo"},
            )
        )
    return [contract], clauses


def main() -> None:
    processed_dir = DATA_DIR / "processed"
    artifacts_dir = PROJECT_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    cuad_zip = DATA_DIR / "raw" / "cuad" / "CUAD_v1.zip"
    contractnli_zip = DATA_DIR / "raw" / "contractnli" / "contract-nli.zip"
    nli_contracts = []
    nli_examples = []
    mode = "demo_fallback"

    if cuad_zip.exists():
        contracts, clause_examples = load_cuad_zip(cuad_zip, clause_types=MVP_CLAUSE_TYPES)
        mode = "raw_cuad"
    else:
        contracts, clause_examples = demo_contracts()

    if contractnli_zip.exists():
        nli_contracts, nli_examples = load_contractnli_zip(contractnli_zip)

    ru_legal_qa_contracts = []
    ru_legal_qa_examples = []
    try:
        ru_legal_qa_contracts, ru_legal_qa_examples = load_ru_legal_qa()
        contracts = contracts + ru_legal_qa_contracts
        clause_examples = clause_examples + ru_legal_qa_examples
    except Exception as exc:
        print(f"Skip Ru-Legal-QA-v1 normalization: {exc}")

    all_contracts = contracts + nli_contracts
    chunks = [
        chunk
        for contract in all_contracts
        for chunk in chunk_contract(contract.contract_id, contract.text, chunk_size_words=800, overlap_words=150)
    ]

    write_jsonl(processed_dir / "contracts.jsonl", all_contracts)
    write_jsonl(processed_dir / "clause_examples.jsonl", clause_examples)
    write_jsonl(processed_dir / "nli_examples.jsonl", nli_examples)
    write_jsonl(processed_dir / "chunks.jsonl", chunks)
    splits: dict[str, list[str]] = {}
    for contract in all_contracts:
        splits.setdefault(contract.split, []).append(contract.contract_id)
    (processed_dir / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    span_examples = [example for example in clause_examples if example.gold_spans]
    (artifacts_dir / "data_summary.json").write_text(
        json.dumps(
            {
                "contracts": len(all_contracts),
                "cuad_and_ru_contracts": len(contracts),
                "contractnli_contracts": len(nli_contracts),
                "ru_legal_qa_contracts": len(ru_legal_qa_contracts),
                "clause_examples": len(clause_examples),
                "ru_legal_qa_examples": len(ru_legal_qa_examples),
                "nli_examples": len(nli_examples),
                "chunks": len(chunks),
                "gold_span_found_rate": len(span_examples) / len(clause_examples) if clause_examples else 0.0,
                "mode": mode,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
