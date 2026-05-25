import json
import os
import re
from html.parser import HTMLParser
from urllib.request import Request, urlopen

from contract_inspector.data.io import write_jsonl
from contract_inspector.data.schemas import ClauseExample, ContractDoc, GoldSpan
from contract_inspector.features.chunking import chunk_contract
from contract_inspector.settings.domain_config import load_domain_terms
from contract_inspector.settings.paths import DATA_DIR, PROJECT_DIR


def build_eis_btk_sample(urls: list[str] | None = None) -> tuple[list[ContractDoc], list[ClauseExample], list[dict]]:
    config = load_domain_terms()["eis_btk"]
    default_urls = [str(url) for url in config.get("default_urls", [])]
    name_prefix = str(config.get("name_prefix", "EIS BTK"))
    contracts = []
    examples = []
    errors = []
    for index, url in enumerate(urls or default_urls, start=1):
        try:
            text = _fetch_text(url)
        except Exception as exc:
            errors.append({"url": url, "error": str(exc)})
            continue
        contract_id = f"eis_btk_{index:04d}"
        contracts.append(
            ContractDoc(
                contract_id=contract_id,
                dataset="eis_btk",
                split="raw",
                source_file=url,
                name=f"{name_prefix} {index}",
                text=text,
                metadata={"source_url": url, "weak_labels": True},
            )
        )
        for clause_type, span in _weak_section_spans(text).items():
            examples.append(
                ClauseExample(
                    example_id=f"{contract_id}__{_slug(clause_type)}",
                    contract_id=contract_id,
                    clause_type=clause_type,
                    label_present=True,
                    answer_value=None,
                    gold_spans=[GoldSpan(start=span[0], end=span[1], text=text[span[0] : span[1]])],
                    metadata={"source": "eis_btk", "weak_label": True, "match_status": "heading_section"},
                )
            )
    return contracts, examples, errors


def write_eis_btk_sample(urls: list[str] | None = None) -> dict:
    default_urls = [str(url) for url in load_domain_terms()["eis_btk"].get("default_urls", [])]
    contracts, examples, errors = build_eis_btk_sample(urls)
    chunks = [
        chunk
        for contract in contracts
        for chunk in chunk_contract(contract.contract_id, contract.text, chunk_size_words=500, overlap_words=100)
    ]
    processed_dir = DATA_DIR / "processed"
    artifacts_dir = PROJECT_DIR / "artifacts"
    write_jsonl(processed_dir / "eis_btk.contracts.jsonl", contracts)
    write_jsonl(processed_dir / "eis_btk.clause_examples.jsonl", examples)
    write_jsonl(processed_dir / "eis_btk.chunks.jsonl", chunks)
    summary = {
        "dataset": "eis_btk",
        "contracts": len(contracts),
        "weak_examples": len(examples),
        "chunks": len(chunks),
        "source_urls": urls or default_urls,
        "errors": errors,
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "eis_btk_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    urls_path = DATA_DIR / "raw" / "eis_btk_urls.txt"
    urls = None
    if urls_path.exists():
        urls = [line.strip() for line in urls_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps(write_eis_btk_sample(urls), indent=2, ensure_ascii=False))


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        data = data.strip()
        if data:
            self.parts.append(data)

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "\n".join(self.parts))


def _fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    timeout = float(os.getenv("EIS_BTK_TIMEOUT_SECONDS", "15"))
    with urlopen(request, timeout=timeout) as response:
        html = response.read().decode("utf-8", errors="replace")
    parser = _TextExtractor()
    parser.feed(html)
    return parser.text()


def _weak_section_spans(text: str) -> dict[str, tuple[int, int]]:
    section_patterns = load_domain_terms()["eis_btk"].get("section_patterns", {})
    starts = []
    for clause_type, patterns in section_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                starts.append((match.start(), clause_type))
                break
    starts.sort()
    spans = {}
    for index, (start, clause_type) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else min(len(text), start + 4000)
        spans[clause_type] = (start, end)
    return spans


def _slug(value: str) -> str:
    return re.sub(r"[^\w]+", "_", value.lower()).strip("_")


if __name__ == "__main__":
    main()
