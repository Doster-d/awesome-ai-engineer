import pytest

from contract_inspector.features.chunking import chunk_contract


def test_short_contract_gives_one_chunk():
    text = "This Agreement shall be governed by the laws of New York."
    chunks = chunk_contract("demo", text, chunk_size_words=20, overlap_words=5)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "demo__chunk_0000"
    assert text[chunks[0].char_start : chunks[0].char_end] == chunks[0].text


def test_long_contract_has_overlap_and_preserves_offsets():
    words = [f"word{i}" for i in range(25)]
    text = "\n  ".join(words)
    chunks = chunk_contract("demo", text, chunk_size_words=10, overlap_words=3)

    assert len(chunks) == 4
    assert chunks[0].char_end > chunks[1].char_start
    for chunk in chunks:
        assert text[chunk.char_start : chunk.char_end] == chunk.text


def test_empty_text_returns_no_chunks():
    assert chunk_contract("demo", "   ") == []


def test_invalid_chunk_parameters_raise():
    with pytest.raises(ValueError):
        chunk_contract("demo", "text", chunk_size_words=0)
    with pytest.raises(ValueError):
        chunk_contract("demo", "text", chunk_size_words=10, overlap_words=10)
