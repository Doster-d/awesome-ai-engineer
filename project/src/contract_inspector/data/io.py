from collections.abc import Iterator
from collections.abc import Iterable
from pathlib import Path

import orjson
from pydantic import BaseModel


def write_jsonl(path: Path, rows: Iterable[BaseModel | dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        for row in rows:
            if isinstance(row, BaseModel):
                data = row.model_dump(mode="json")
            else:
                data = row

            f.write(orjson.dumps(data))
            f.write(b"\n")


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            yield orjson.loads(line)
