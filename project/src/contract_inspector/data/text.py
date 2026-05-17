import re
import math
import json
from pathlib import Path
from typing import Any
import pandas as pd


CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
SPACES_RE = re.compile(r"[ \t]+")
NEWLINES_RE = re.compile(r"\n{3,}")
WORD_RE = re.compile(r"\b\S+\b")
ANSWER_COLUMN_RE = re.compile(r"\s*-\s*Answer\s*$", re.IGNORECASE)

EMPTY_STRINGS = {"", "nan", "NaN", "None", "none", "null", "NULL", "[]", "{}"}


def normalize_text(text: str) -> str:
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHARS_RE.sub("", text)
    text = SPACES_RE.sub(" ", text)
    text = NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def count_words(text: str) -> int:
    return len(WORD_RE.findall(str(text)))


def approx_tokens_en(text: str) -> int:
    # 1 token ~ 4 characters in English
    return math.ceil(len(str(text)) / 4)


def is_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False

    text = str(value).strip()
    return text not in EMPTY_STRINGS


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")
