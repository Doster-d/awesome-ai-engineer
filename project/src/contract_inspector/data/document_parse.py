from typing import Any
import pandas as pd
import re
from contract_inspector.data.text import normalize_text, count_words, approx_tokens_en

def normalize_doc_key(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"\.(pdf|txt)$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.lower()

def contractnli_label_text(label_payload: Any) -> str:
    if isinstance(label_payload, dict):
        return str(label_payload.get("hypothesis") or label_payload.get("short_description") or label_payload)

    return str(label_payload)


def contractnli_label_short_description(label_payload: Any) -> str:
    if isinstance(label_payload, dict):
        return str(label_payload.get("short_description") or "")

    return ""


def normalize_contractnli_choice(value: Any) -> str:
    text = str(value).strip().lower()

    mapping = {
        "entailment": "entailment",
        "entailed": "entailment",
        "contradiction": "contradiction",
        "contradicted": "contradiction",
        "notmentioned": "not_mentioned",
        "not_mentioned": "not_mentioned",
        "not mentioned": "not_mentioned",
        "neutral": "not_mentioned",
    }

    return mapping.get(text, text)


def normalize_span(span: Any, text: str, span_id: str | int | None = None) -> dict[str, Any] | None:
    if isinstance(span, dict):
        start = span.get("start")
        end = span.get("end")

        if start is None or end is None:
            return None

        start = int(start)
        end = int(end)

        return {
            "span_id": str(span.get("id") or span_id or ""),
            "start": start,
            "end": end,
            "text": span.get("text") or text[start:end],
        }

    if isinstance(span, (list, tuple)) and len(span) >= 2:
        start = int(span[0])
        end = int(span[1])

        return {
            "span_id": str(span_id or ""),
            "start": start,
            "end": end,
            "text": text[start:end],
        }

    return None


def build_contractnli_span_lookup(doc: dict[str, Any], text: str) -> dict[str, dict[str, Any]]:
    spans = doc.get("spans") or []
    lookup: dict[str, dict[str, Any]] = {}

    if isinstance(spans, dict):
        span_items = spans.items()
    elif isinstance(spans, list):
        span_items = enumerate(spans)
    else:
        span_items = []

    for span_id, span in span_items:
        normalized = normalize_span(span, text, span_id=span_id)
        if normalized is not None:
            lookup[str(span_id)] = normalized

            internal_id = normalized.get("span_id")
            if internal_id:
                lookup[str(internal_id)] = normalized

    return lookup


def extract_evidence_from_annotation(
    annotation: dict[str, Any],
    span_lookup: dict[str, dict[str, Any]],
    text: str,
) -> list[dict[str, Any]]:
    evidence_candidates = None

    for key in ["spans", "evidence", "evidence_spans", "evidences"]:
        if key in annotation and annotation[key] is not None:
            evidence_candidates = annotation[key]
            break

    if evidence_candidates is None:
        return []

    if isinstance(evidence_candidates, (str, int, dict)):
        evidence_candidates = [evidence_candidates]

    if not isinstance(evidence_candidates, list):
        return []

    evidence = []

    for item in evidence_candidates:
        parsed = None

        if isinstance(item, (str, int)):
            parsed = span_lookup.get(str(item))

        elif isinstance(item, dict):
            if "start" in item and "end" in item:
                parsed = normalize_span(item, text)
            else:
                span_id = item.get("id") or item.get("span_id")
                if span_id is not None:
                    parsed = span_lookup.get(str(span_id))

        if parsed is not None:
            evidence.append(parsed)

    return evidence


def extract_annotation_choice(annotation: dict[str, Any]) -> Any:
    for key in ["choice", "label", "classification", "answer", "value"]:
        if key in annotation:
            return annotation[key]

    return None


def iter_contractnli_annotations(doc: dict[str, Any]):
    annotation_sets = doc.get("annotation_sets") or []

    if isinstance(annotation_sets, dict):
        annotation_set_items = annotation_sets.items()
    elif isinstance(annotation_sets, list):
        annotation_set_items = enumerate(annotation_sets)
    else:
        return

    for annotation_set_id, annotation_set in annotation_set_items:
        if not isinstance(annotation_set, dict):
            continue

        annotations = annotation_set.get("annotations")

        if isinstance(annotations, dict):
            for label_id, annotation in annotations.items():
                if isinstance(annotation, dict):
                    yield str(annotation_set_id), str(label_id), annotation
            continue

        for label_id, annotation in annotation_set.items():
            if label_id in {"id", "annotator", "created_at", "updated_at"}:
                continue

            if isinstance(annotation, dict):
                yield str(annotation_set_id), str(label_id), annotation


def flatten_contractnli_split(split: str, payload: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = payload.get("labels") or {}
    documents = payload.get("documents") or []

    if not isinstance(documents, list):
        raise TypeError(f"Expected documents as list, got: {type(documents)}")

    label_hypothesis_by_id = {
        str(label_id): contractnli_label_text(label_payload)
        for label_id, label_payload in labels.items()
    }

    label_short_by_id = {
        str(label_id): contractnli_label_short_description(label_payload)
        for label_id, label_payload in labels.items()
    }

    doc_rows = []
    example_rows = []

    for doc_index, doc in enumerate(documents):
        if not isinstance(doc, dict):
            continue

        doc_id = str(doc.get("id") or f"{split}_{doc_index:06d}")
        file_name = str(doc.get("file_name") or doc.get("filename") or doc_id)
        text = normalize_text(str(doc.get("text") or ""))

        span_lookup = build_contractnli_span_lookup(doc, text)

        annotation_count = 0
        annotation_set_ids = set()

        for annotation_set_id, label_id, annotation in iter_contractnli_annotations(doc):
            annotation_count += 1
            annotation_set_ids.add(annotation_set_id)

            raw_choice = extract_annotation_choice(annotation)
            gold_label = normalize_contractnli_choice(raw_choice)

            evidence = extract_evidence_from_annotation(annotation, span_lookup, text)
            evidence_total_chars = sum(
                int(item["end"]) - int(item["start"])
                for item in evidence
                if item.get("start") is not None and item.get("end") is not None
            )

            example_rows.append(
                {
                    "dataset": "contractnli",
                    "split": split,
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "document_type": doc.get("document_type"),
                    "url": doc.get("url"),
                    "annotation_set_id": annotation_set_id,
                    "label_id": label_id,
                    "label_short_description": label_short_by_id.get(label_id, ""),
                    "hypothesis": label_hypothesis_by_id.get(label_id, label_id),
                    "gold_label_raw": raw_choice,
                    "gold_label": gold_label,
                    "evidence_count": len(evidence),
                    "evidence_total_chars": evidence_total_chars,
                    "evidence": evidence,
                }
            )

        doc_rows.append(
            {
                "dataset": "contractnli",
                "split": split,
                "doc_id": doc_id,
                "file_name": file_name,
                "document_type": doc.get("document_type"),
                "url": doc.get("url"),
                "chars": len(text),
                "words": count_words(text),
                "approx_tokens": approx_tokens_en(text),
                "spans_count": len(span_lookup),
                "annotation_sets_count": len(annotation_set_ids),
                "annotations_count": annotation_count,
                "empty": not bool(text),
            }
        )

    return pd.DataFrame(doc_rows), pd.DataFrame(example_rows)
