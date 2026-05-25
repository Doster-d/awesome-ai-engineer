import hashlib
from pathlib import Path

import joblib
import pandas as pd

from contract_inspector.features.chunking import chunk_contract
from contract_inspector.features.clause_queries import build_clause_query
from contract_inspector.features.retrieval_features import build_retrieval_feature_row
from contract_inspector.retrieval.bm25 import BM25ChunkRetriever
from contract_inspector.retrieval.schemas import RetrievalHit
from contract_inspector.retrieval.tfidf import TfidfChunkRetriever
from contract_inspector.rlm.navigator import RLMNavigator
from contract_inspector.rlm.tools import RuntimeDocumentTools
from contract_inspector.settings.domain_config import load_domain_terms
from contract_inspector.settings.paths import PROJECT_DIR


DEFAULT_CLAUSE_TYPES = ["Governing Law", "Cap On Liability", "Anti-Assignment"]


class ContractInspectionPipeline:
    model_version = "bm25_tfidf_ranker_runtime_v1"

    def __init__(self, ranker_path: Path | None = None) -> None:
        self.ranker_path = ranker_path or PROJECT_DIR / "artifacts" / "feature_ranker.joblib"
        self.ranker = joblib.load(self.ranker_path) if self.ranker_path.exists() else None

    def inspect(
        self,
        contract_text: str,
        clause_types: list[str],
        top_k: int = 5,
        use_rlm: bool = False,
    ) -> dict:
        contract_id = self._runtime_contract_id(contract_text)
        clauses = clause_types or DEFAULT_CLAUSE_TYPES
        chunks = chunk_contract(
            contract_id=contract_id,
            text=contract_text,
            chunk_size_words=220,
            overlap_words=50,
        )
        bm25 = BM25ChunkRetriever().fit(chunks)
        tfidf = TfidfChunkRetriever().fit(chunks)
        rlm_tools = RuntimeDocumentTools(chunks) if use_rlm else None
        rlm_navigator = RLMNavigator(rlm_tools) if rlm_tools else None

        results = []
        for clause_type in clauses:
            query = build_clause_query(clause_type)
            hits = self._candidate_hits(
                query=query,
                clause_type=clause_type,
                chunks=chunks,
                bm25=bm25,
                tfidf=tfidf,
                contract_id=contract_id,
                top_k=top_k,
            )
            evidence = [
                {
                    "chunk_id": hit.chunk_id,
                    "char_start": hit.char_start,
                    "char_end": hit.char_end,
                    "quote": hit.text,
                    "score": hit.score,
                    "source": "bm25+tfidf+ranker" if self.ranker else "bm25+tfidf",
                }
                for hit in hits
                if hit.score > 0
            ]
            confidence = evidence[0]["score"] if evidence else 0.0
            found = bool(evidence)
            answer = evidence[0]["quote"] if found else "Not found"
            risk_level = "unknown"
            rlm_trace = []

            if use_rlm and rlm_navigator:
                rlm_result = rlm_navigator.inspect_clause(
                    clause_type=clause_type,
                    contract_id=contract_id,
                    seed_hits=hits,
                    top_k=min(top_k, 10),
                )
                rlm_trace = rlm_result.get("trace", [])
                if rlm_result.get("enabled") and isinstance(rlm_result.get("answer"), dict):
                    structured = rlm_result["answer"]
                    found = bool(structured.get("found", found))
                    answer = str(structured.get("answer", answer))
                    confidence = _coerce_confidence(structured.get("confidence", confidence), confidence)
                    risk_level = structured.get("risk_level", risk_level)

            results.append(
                {
                    "clause_type": clause_type,
                    "found": found,
                    "answer": answer,
                    "evidence": evidence,
                    "confidence": confidence,
                    "risk_level": risk_level,
                    "rlm_trace": rlm_trace,
                }
            )

        return {
            "contract_id": contract_id,
            "results": results,
            "model_version": self.model_version if not use_rlm else f"{self.model_version}_rlm_fallback",
        }

    @staticmethod
    def _runtime_contract_id(contract_text: str) -> str:
        digest = hashlib.sha1(contract_text.encode("utf-8")).hexdigest()[:12]
        return f"runtime_{digest}"

    def _candidate_hits(
        self,
        query: str,
        clause_type: str,
        chunks: list,
        bm25: BM25ChunkRetriever,
        tfidf: TfidfChunkRetriever,
        contract_id: str,
        top_k: int,
    ) -> list[RetrievalHit]:
        bm25_hits = bm25.search(query=query, contract_id=contract_id, top_k=max(top_k, 10))
        tfidf_hits = tfidf.search(query=query, contract_id=contract_id, top_k=max(top_k, 10))
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        by_chunk: dict[str, dict] = {}
        for hit in tfidf_hits:
            by_chunk.setdefault(hit.chunk_id, {})["tfidf_hit"] = hit
        for hit in bm25_hits:
            by_chunk.setdefault(hit.chunk_id, {})["bm25_hit"] = hit

        if not self.ranker:
            merged = [data.get("bm25_hit") or data["tfidf_hit"] for data in by_chunk.values()]
            return [
                hit.model_copy(update={"rank": rank})
                for rank, hit in enumerate(sorted(merged, key=lambda item: item.score, reverse=True)[:top_k], start=1)
            ]

        rows = []
        hits = []
        for chunk_id, data in by_chunk.items():
            chunk = chunks_by_id[chunk_id]
            bm25_hit = data.get("bm25_hit")
            tfidf_hit = data.get("tfidf_hit")
            base_hit = bm25_hit or tfidf_hit
            row = build_retrieval_feature_row(query, clause_type, chunk, base_hit, is_relevant=None)
            row.update(
                {
                    "tfidf_score": tfidf_hit.score if tfidf_hit else 0.0,
                    "bm25_score": bm25_hit.score if bm25_hit else 0.0,
                    "tfidf_rank": tfidf_hit.rank if tfidf_hit else max(top_k, 10) + 1,
                    "bm25_rank": bm25_hit.rank if bm25_hit else max(top_k, 10) + 1,
                    "relative_position": chunk.chunk_index / max(1, len(chunks) - 1),
                }
            )
            rows.append(row)
            hits.append(base_hit)

        feature_frame = pd.DataFrame(rows)
        feature_columns = [
            "tfidf_score",
            "bm25_score",
            "tfidf_rank",
            "bm25_rank",
            "query_term_coverage",
            "matched_query_terms_count",
            "legal_marker_count",
            "word_count",
            "relative_position",
            "chunk_index",
            "number_count",
            "date_count",
            "money_count",
            "clause_type",
        ]
        scores = self.ranker.predict_proba(feature_frame[feature_columns])[:, 1]
        ranked = sorted(zip(hits, scores, strict=True), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            hit.model_copy(update={"score": float(score), "rank": rank})
            for rank, (hit, score) in enumerate(ranked, start=1)
        ]


def _coerce_confidence(value: object, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        labels = {str(key): float(score) for key, score in load_domain_terms().get("confidence_labels", {}).items()}
        if normalized in labels:
            return labels[normalized]
        try:
            return float(normalized)
        except ValueError:
            return default
    return default
