from contract_inspector.retrieval.evaluate import recall_at_k, reciprocal_rank
from contract_inspector.retrieval.schemas import RetrievalHit


def ranking_metrics(hits: list[RetrievalHit], relevant_chunk_ids: set[str]) -> dict:
    return {
        "recall@1": recall_at_k(hits, relevant_chunk_ids, 1),
        "recall@3": recall_at_k(hits, relevant_chunk_ids, 3),
        "recall@5": recall_at_k(hits, relevant_chunk_ids, 5),
        "recall@10": recall_at_k(hits, relevant_chunk_ids, 10),
        "mrr": reciprocal_rank(hits, relevant_chunk_ids),
    }
