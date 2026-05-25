from pathlib import Path

from contract_inspector.data.io import read_jsonl, write_jsonl
from contract_inspector.data.schemas import ContractDoc


def test_jsonl_roundtrip(tmp_path: Path):
    path = tmp_path / "contracts.jsonl"
    doc = ContractDoc(
        contract_id="cuad_0001",
        dataset="cuad",
        split="train",
        source_file="example.txt",
        name="Example",
        text="This is a test contract.",
    )

    write_jsonl(path, [doc, {"contract_id": "manual"}])
    rows = list(read_jsonl(path))

    assert rows[0]["contract_id"] == "cuad_0001"
    assert rows[1]["contract_id"] == "manual"


def test_read_jsonl_skips_empty_lines(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"x":1}\n\n{"x":2}\n')

    assert [row["x"] for row in read_jsonl(path)] == [1, 2]
