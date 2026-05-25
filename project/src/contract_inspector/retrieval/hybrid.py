from contract_inspector.retrieval.schemas import RetrievalHit


def reciprocal_rank_fusion(
    hit_lists: list[list[RetrievalHit]],
    k: int = 60,
    top_k: int = 10,
) -> list[RetrievalHit]:
    by_chunk: dict[str, RetrievalHit] = {}
    scores: dict[str, float] = {}
    for hits in hit_lists:
        for hit in hits:
            by_chunk.setdefault(hit.chunk_id, hit)
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + hit.rank)

    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    fused = []
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        hit = by_chunk[chunk_id].model_copy(update={"score": scores[chunk_id], "rank": rank})
        hit.metadata["retriever"] = "rrf_hybrid"
        fused.append(hit)
    return fused
