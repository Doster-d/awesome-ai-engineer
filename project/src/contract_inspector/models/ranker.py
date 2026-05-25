from contract_inspector.retrieval.schemas import RetrievalHit


class RuleBasedRanker:
    def rerank(self, hits: list[RetrievalHit], top_k: int | None = None) -> list[RetrievalHit]:
        ranked = sorted(hits, key=lambda hit: hit.score, reverse=True)
        limit = top_k or len(ranked)
        return [hit.model_copy(update={"rank": rank}) for rank, hit in enumerate(ranked[:limit], start=1)]
