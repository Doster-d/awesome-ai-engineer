import json
import zipfile
from collections.abc import Iterable
from pathlib import Path

from contract_inspector.data.schemas import ContractDoc, GoldSpan, NLIExample


def normalize_contractnli_rows(rows: Iterable[dict]) -> tuple[list[ContractDoc], list[NLIExample]]:
    contracts: list[ContractDoc] = []
    examples: list[NLIExample] = []
    for index, row in enumerate(rows, start=1):
        contract_id = row.get("contract_id") or f"contractnli_{index:04d}"
        text = row.get("text") or row.get("contract_text") or ""
        label = row.get("gold_label") or row.get("label") or "not_mentioned"
        spans = [
            GoldSpan(start=span["start"], end=span["end"], text=span["text"])
            for span in row.get("gold_spans", [])
            if label != "not_mentioned"
        ]
        contracts.append(
            ContractDoc(
                contract_id=contract_id,
                dataset="contractnli",
                split=row.get("split", "train"),
                source_file=row.get("source_file"),
                name=row.get("name"),
                text=text,
            )
        )
        examples.append(
            NLIExample(
                example_id=row.get("example_id") or f"{contract_id}__hyp_{index:04d}",
                contract_id=contract_id,
                hypothesis_id=row.get("hypothesis_id") or f"hyp_{index:04d}",
                hypothesis=row.get("hypothesis", ""),
                gold_label=label,
                gold_spans=spans,
            )
        )
    return contracts, examples


def load_contractnli_zip(path: Path) -> tuple[list[ContractDoc], list[NLIExample]]:
    contracts: list[ContractDoc] = []
    examples: list[NLIExample] = []

    with zipfile.ZipFile(path) as archive:
        for split in ["train", "dev", "test"]:
            payload = json.loads(archive.read(f"contract-nli/{split}.json"))
            labels = payload["labels"]
            for document in payload["documents"]:
                contract_id = f"contractnli_{split}_{document['id']}"
                text = document["text"]
                contracts.append(
                    ContractDoc(
                        contract_id=contract_id,
                        dataset="contractnli",
                        split=split,
                        source_file=document.get("file_name"),
                        name=document.get("file_name"),
                        text=text,
                        metadata={"document_type": document.get("document_type"), "url": document.get("url")},
                    )
                )
                span_offsets = document.get("spans", [])
                annotations = document.get("annotation_sets", [{}])[0].get("annotations", {})
                for hypothesis_id, annotation in annotations.items():
                    label = _normalize_label(annotation["choice"])
                    gold_spans = []
                    if label != "not_mentioned":
                        for span_index in annotation.get("spans", []):
                            start, end = span_offsets[span_index]
                            gold_spans.append(GoldSpan(start=start, end=end, text=text[start:end]))
                    examples.append(
                        NLIExample(
                            example_id=f"{contract_id}__{hypothesis_id}",
                            contract_id=contract_id,
                            hypothesis_id=hypothesis_id,
                            hypothesis=labels[hypothesis_id]["hypothesis"],
                            gold_label=label,
                            gold_spans=gold_spans,
                            metadata={"short_description": labels[hypothesis_id]["short_description"]},
                        )
                    )

    return contracts, examples


def _normalize_label(label: str) -> str:
    mapping = {
        "Entailment": "entailment",
        "Contradiction": "contradiction",
        "NotMentioned": "not_mentioned",
    }
    return mapping.get(label, label.lower())
