import re

from contract_inspector.data.schemas import ChunkRecord


def chunk_contract(
    contract_id: str,
    text: str,
    chunk_size_words: int = 800,
    overlap_words: int = 150,
) -> list[ChunkRecord]:
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    if overlap_words < 0:
        raise ValueError("overlap_words must be non-negative")
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")

    word_matches = list(re.finditer(r"\S+", text))
    if not word_matches:
        return []

    chunks: list[ChunkRecord] = []
    step = chunk_size_words - overlap_words
    chunk_index = 0

    for word_start in range(0, len(word_matches), step):
        window = word_matches[word_start : word_start + chunk_size_words]
        if not window:
            break

        char_start = window[0].start()
        char_end = window[-1].end()
        chunk_text = text[char_start:char_end]

        chunks.append(
            ChunkRecord(
                chunk_id=f"{contract_id}__chunk_{chunk_index:04d}",
                contract_id=contract_id,
                chunk_index=chunk_index,
                char_start=char_start,
                char_end=char_end,
                text=chunk_text,
                word_count=len(window),
            )
        )

        chunk_index += 1
        if word_start + chunk_size_words >= len(word_matches):
            break

    return chunks
