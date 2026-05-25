import json
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path

from contract_inspector.data.schemas import ClauseExample, ContractDoc, GoldSpan


MVP_CLAUSE_TYPES = [
    "Governing Law",
    "Cap On Liability",
    "Anti-Assignment",
    "License Grant",
    "Audit Rights",
    "Termination For Convenience",
    "Insurance",
    "Change Of Control",
    "Non-Compete",
]


def find_gold_span(contract_text: str, evidence_text: str) -> GoldSpan | None:
    if not evidence_text:
        return None
    start = contract_text.find(evidence_text)
    if start == -1:
        return None
    end = start + len(evidence_text)
    return GoldSpan(start=start, end=end, text=contract_text[start:end])


def normalize_cuad_rows(rows: Iterable[dict]) -> tuple[list[ContractDoc], list[ClauseExample]]:
    contracts: list[ContractDoc] = []
    examples: list[ClauseExample] = []
    for index, row in enumerate(rows, start=1):
        contract_id = row.get("contract_id") or f"cuad_{index:04d}"
        text = row.get("text") or row.get("contract_text") or ""
        contracts.append(
            ContractDoc(
                contract_id=contract_id,
                dataset="cuad",
                split=row.get("split", "train"),
                source_file=row.get("source_file"),
                name=row.get("name"),
                text=text,
                metadata={k: v for k, v in row.items() if k not in {"text", "contract_text"}},
            )
        )
        for clause_type in MVP_CLAUSE_TYPES:
            answer = row.get(f"{clause_type}-Answer")
            evidence_text = row.get(clause_type) or row.get(f"{clause_type}-Clause") or ""
            span = find_gold_span(text, evidence_text)
            examples.append(
                ClauseExample(
                    example_id=f"{contract_id}__{clause_type.lower().replace(' ', '_').replace('-', '_')}",
                    contract_id=contract_id,
                    clause_type=clause_type,
                    label_present=bool(answer),
                    answer_value=answer,
                    gold_spans=[span] if span else [],
                    metadata={"match_status": "exact" if span else "missing"},
                )
            )
    return contracts, examples


def load_cuad_zip(path: Path, clause_types: list[str] | None = None) -> tuple[list[ContractDoc], list[ClauseExample]]:
    selected = set(clause_types or MVP_CLAUSE_TYPES)
    with zipfile.ZipFile(path) as archive:
        payload = json.loads(archive.read("CUAD_v1/CUAD_v1.json"))

    contracts: list[ContractDoc] = []
    examples: list[ClauseExample] = []
    for doc_index, item in enumerate(payload["data"], start=1):
        title = item["title"]
        contract_id = f"cuad_{doc_index:04d}"
        paragraph = item["paragraphs"][0]
        text = paragraph["context"]
        contracts.append(
            ContractDoc(
                contract_id=contract_id,
                dataset="cuad",
                split="train",
                source_file=title,
                name=title,
                text=text,
                metadata={"title": title},
            )
        )

        for qa in paragraph["qas"]:
            clause_type = _clause_type_from_question(qa["question"])
            if clause_type not in selected:
                continue

            gold_spans = [
                GoldSpan(
                    start=answer["answer_start"],
                    end=answer["answer_start"] + len(answer["text"]),
                    text=answer["text"],
                )
                for answer in qa.get("answers", [])
            ]
            examples.append(
                ClauseExample(
                    example_id=f"{contract_id}__{_slug(clause_type)}",
                    contract_id=contract_id,
                    clause_type=clause_type,
                    label_present=not qa.get("is_impossible", False) and bool(gold_spans),
                    answer_value="Yes" if gold_spans else "No",
                    gold_spans=gold_spans,
                    metadata={
                        "qa_id": qa["id"],
                        "match_status": "provided_offset" if gold_spans else "no_evidence",
                    },
                )
            )

    return contracts, examples


def _clause_type_from_question(question: str) -> str:
    match = re.search(r'related to "(.+?)"', question)
    if match:
        return match.group(1)
    return question


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
