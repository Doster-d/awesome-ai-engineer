from contract_inspector.data.schemas import ChunkRecord, GoldSpan
from contract_inspector.retrieval.evaluate import (
    is_relevant_chunk,
    reciprocal_rank,
    recall_at_k,
    span_overlap,
    span_overlap_ratio,
)
from contract_inspector.retrieval.schemas import RetrievalHit


def test_span_overlap_metrics():
    assert span_overlap(10, 20, 15, 30) == 5
    assert span_overlap_ratio(10, 20, 15, 30) == 0.5
    assert span_overlap_ratio(10, 10, 0, 20) == 0.0


def test_relevant_chunk_by_gold_overlap():
    chunk = ChunkRecord(
        chunk_id="c1",
        contract_id="doc",
        chunk_index=0,
        char_start=100,
        char_end=200,
        text="x",
        word_count=1,
    )
    assert is_relevant_chunk(chunk, [GoldSpan(start=150, end=180, text="gold")])


def test_retrieval_metrics():
    hits = [
        RetrievalHit(query="q", chunk_id="a", contract_id="d", score=1, rank=1, text="", char_start=0, char_end=1),
        RetrievalHit(query="q", chunk_id="b", contract_id="d", score=0.5, rank=2, text="", char_start=1, char_end=2),
    ]
    assert recall_at_k(hits, {"b"}, 1) == 0.0
    assert recall_at_k(hits, {"b"}, 2) == 1.0
    assert reciprocal_rank(hits, {"b"}) == 0.5
