from contract_inspector.features.chunking import chunk_contract
from contract_inspector.retrieval.bm25 import BM25ChunkRetriever
from contract_inspector.retrieval.tfidf import TfidfChunkRetriever


def test_retrievers_find_clause_chunk_within_contract():
    text = (
        "Intro words. " * 30
        + "This Agreement shall be governed by the laws of New York. "
        + "Operational appendix. " * 30
    )
    chunks = chunk_contract("doc1", text, chunk_size_words=20, overlap_words=5)

    bm25_hits = BM25ChunkRetriever().fit(chunks).search("governing law governed by laws", "doc1", top_k=3)
    tfidf_hits = TfidfChunkRetriever().fit(chunks).search("governing law governed by laws", "doc1", top_k=3)

    assert bm25_hits
    assert tfidf_hits
    assert "governed by the laws" in bm25_hits[0].text
    assert "governed by the laws" in tfidf_hits[0].text
