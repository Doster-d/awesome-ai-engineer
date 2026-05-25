import re


def has_section_like_title(text: str) -> bool:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return bool(re.match(r"^(\d+(\.\d+)*\.?|section\s+\d+|article\s+[ivx\d]+)", first_line, re.I))
