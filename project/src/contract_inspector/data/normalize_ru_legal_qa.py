import json
import re
from collections.abc import Iterable

from datasets import load_dataset

from contract_inspector.data.io import read_jsonl, write_jsonl
from contract_inspector.data.schemas import ClauseExample, ContractDoc, GoldSpan
from contract_inspector.features.chunking import chunk_contract
from contract_inspector.settings.domain_config import load_domain_terms
from contract_inspector.settings.paths import DATA_DIR, PROJECT_DIR


DATASET_NAME = "Roflmax/Ru-Legal-QA-v1"


def load_ru_legal_qa(limit: int | None = None) -> tuple[list[ContractDoc], list[ClauseExample]]:
    rows = load_dataset(DATASET_NAME, split="train")
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    return normalize_ru_legal_qa_rows(rows)


def normalize_ru_legal_qa_rows(rows: Iterable[dict]) -> tuple[list[ContractDoc], list[ClauseExample]]:
    config = load_domain_terms()["ru_legal_qa"]
    contracts_by_id: dict[str, ContractDoc] = {}
    examples: list[ClauseExample] = []

    for row_index, row in enumerate(rows, start=1):
        full_text = _ensure_obj(_pick(row, *config["full_text_fields"], default={}))
        law_quotes = _ensure_obj(_pick(row, *config["law_quotes_fields"], default={}))
        question = str(_pick(row, *config["question_fields"], default=""))
        answer = str(_pick(row, *config["answer_fields"], default=""))
        category = str(_pick(row, *config["category_fields"], default=config["default_category"]))

        for law_key, text in _iter_texts(full_text):
            contract_id = f"ru_legal_qa__{_slug(law_key)}"
            if contract_id not in contracts_by_id:
                contracts_by_id[contract_id] = ContractDoc(
                    contract_id=contract_id,
                    dataset="ru_legal_qa_v1",
                    split="test",
                    source_file=DATASET_NAME,
                    name=law_key,
                    text=text,
                    metadata={"language": "ru", "source_dataset": DATASET_NAME},
                )

            spans = []
            for quote in _quotes_for_law(law_quotes, law_key):
                match = _find_quote(text, quote)
                if match:
                    start, end = match
                    spans.append(GoldSpan(start=start, end=end, text=text[start:end]))

            if spans:
                examples.append(
                    ClauseExample(
                        example_id=f"ru_legal_qa_{row_index:04d}__{_slug(law_key)}",
                        contract_id=contract_id,
                        clause_type=category or str(config["default_category"]),
                        label_present=True,
                        answer_value=answer,
                        gold_spans=spans,
                        metadata={
                            "question": question,
                            "category": category,
                            "source_dataset": DATASET_NAME,
                            "match_status": "exact_or_normalized",
                        },
                    )
                )

    return list(contracts_by_id.values()), examples


def write_ru_legal_qa(limit: int | None = None) -> dict:
    contracts, examples = load_ru_legal_qa(limit=limit)
    chunks = [
        chunk
        for contract in contracts
        for chunk in chunk_contract(contract.contract_id, contract.text, chunk_size_words=500, overlap_words=100)
    ]
    processed_dir = DATA_DIR / "processed"
    artifacts_dir = PROJECT_DIR / "artifacts"
    write_jsonl(processed_dir / "ru_legal_qa.contracts.jsonl", contracts)
    write_jsonl(processed_dir / "ru_legal_qa.clause_examples.jsonl", examples)
    write_jsonl(processed_dir / "ru_legal_qa.chunks.jsonl", chunks)
    summary = {
        "dataset": "ru_legal_qa_v1",
        "contracts": len(contracts),
        "examples": len(examples),
        "chunks": len(chunks),
        "source": DATASET_NAME,
        "real_dataset": True,
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "ru_legal_qa_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def load_processed_ru_legal_qa() -> tuple[list[ContractDoc], list[ClauseExample]]:
    contracts_path = DATA_DIR / "processed" / "ru_legal_qa.contracts.jsonl"
    examples_path = DATA_DIR / "processed" / "ru_legal_qa.clause_examples.jsonl"
    if not contracts_path.exists() or not examples_path.exists():
        write_ru_legal_qa()
    return (
        [ContractDoc.model_validate(row) for row in read_jsonl(contracts_path)],
        [ClauseExample.model_validate(row) for row in read_jsonl(examples_path)],
    )


def main() -> None:
    print(json.dumps(write_ru_legal_qa(), indent=2, ensure_ascii=False))


def _pick(row: dict, *names: str, default=None):
    for name in names:
        if name in row:
            return row[name]
    return default


def _ensure_obj(value):
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _iter_texts(full_text) -> Iterable[tuple[str, str]]:
    if isinstance(full_text, dict):
        for key, value in full_text.items():
            if value:
                yield str(key), str(value)
    elif isinstance(full_text, list):
        for index, value in enumerate(full_text):
            if value:
                yield f"document_{index}", str(value)
    elif isinstance(full_text, str) and full_text.strip():
        yield "document", full_text


def _quotes_for_law(law_quotes, law_key: str) -> list[str]:
    if isinstance(law_quotes, dict):
        candidates = law_quotes.get(law_key, [])
        if isinstance(candidates, str):
            return [candidates]
        if isinstance(candidates, list):
            return [str(item) for item in candidates if item]
        return []
    if isinstance(law_quotes, list):
        return [str(item) for item in law_quotes if item]
    if isinstance(law_quotes, str) and law_quotes.strip():
        return [law_quotes]
    return []


def _find_quote(text: str, quote: str) -> tuple[int, int] | None:
    quote = quote.strip()
    if not quote:
        return None
    start = text.find(quote)
    if start >= 0:
        return start, start + len(quote)

    normalized_text = re.sub(r"\s+", " ", text)
    normalized_quote = re.sub(r"\s+", " ", quote)
    start = normalized_text.find(normalized_quote)
    if start < 0:
        return None

    pattern = re.escape(normalized_quote).replace(r"\ ", r"\s+")
    match = re.search(pattern, text)
    if not match:
        return None
    return match.start(), match.end()


def _slug(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^\w]+", "_", value)
    return value.strip("_")[:80] or "document"


if __name__ == "__main__":
    main()
