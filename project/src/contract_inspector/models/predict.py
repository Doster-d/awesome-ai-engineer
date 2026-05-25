from contract_inspector.models.ranker import RuleBasedRanker
from contract_inspector.retrieval.schemas import RetrievalHit


def rerank_hits(hits: list[RetrievalHit], top_k: int | None = None) -> list[RetrievalHit]:
    return RuleBasedRanker().rerank(hits, top_k=top_k)
