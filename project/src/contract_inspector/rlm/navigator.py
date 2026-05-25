import json
from pathlib import Path
import re

from contract_inspector.features.clause_queries import build_clause_query
from contract_inspector.retrieval.schemas import RetrievalHit
from contract_inspector.rlm.config import configure_dspy_from_env
from contract_inspector.rlm.programs import ClauseEvidenceProgram
from contract_inspector.rlm.tools import RuntimeDocumentTools
from contract_inspector.settings.paths import PROJECT_DIR


class RLMNavigator:
    def __init__(
        self,
        tools: RuntimeDocumentTools,
        max_tool_calls: int = 10,
        compiled_program_path: Path | None = None,
    ) -> None:
        self.tools = tools
        self.max_tool_calls = max_tool_calls
        self.compiled_program_path = compiled_program_path or PROJECT_DIR / "artifacts" / "rlm_gepa_program.json"
        self.lm_available = configure_dspy_from_env() is not None
        self.program = ClauseEvidenceProgram()
        if self.lm_available and self.compiled_program_path.exists():
            self.program.load(str(self.compiled_program_path))

    def inspect_clause(
        self,
        clause_type: str,
        contract_id: str,
        seed_hits: list[RetrievalHit] | None = None,
        top_k: int = 3,
    ) -> dict:
        query = build_clause_query(clause_type)
        hits = seed_hits or self.tools.search_chunks(query, contract_id=contract_id, top_k=top_k)
        candidates = []
        chunks_read = []
        for hit in hits[:top_k]:
            chunk = self.tools.read_chunk(hit.chunk_id)
            chunks_read.append(chunk.chunk_id)
            neighbors = self.tools.get_neighbors(chunk.chunk_id, left=1, right=1)
            context = "\n\n".join(neighbor.text for neighbor in neighbors)
            quote = _focused_quote(hit.text, query)
            candidates.append(
                {
                    "chunk_id": hit.chunk_id,
                    "quote": quote,
                    "context": context,
                    "char_start": hit.char_start,
                    "char_end": hit.char_end,
                    "score": hit.score,
                }
            )

        if not self.lm_available:
            return {
                "enabled": False,
                "fallback_reason": "OPENROUTER_API_KEY is not set",
                "clause_type": clause_type,
                "queries": [query],
                "chunks_read": chunks_read,
                "trace": self.tools.trace,
            }

        prediction = self.program(clause_type=clause_type, evidence_quotes=json.dumps(candidates, ensure_ascii=False))
        answer = _safe_json(prediction.answer_json)

        return {
            "enabled": True,
            "clause_type": clause_type,
            "queries": [query],
            "chunks_read": chunks_read,
            "answer": answer,
            "judge_outputs": _safe_json(prediction.judge_outputs, default=[]),
            "trace": self.tools.trace,
        }


def _safe_json(value: str, default: object | None = None) -> object:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default if default is not None else {"raw": value}


def _focused_quote(text: str, query: str, max_chars: int = 900) -> str:
    terms = {term for term in re.findall(r"\w+", query.lower()) if len(term) >= 5}
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return text[:max_chars].strip()

    def score(paragraph: str) -> int:
        lowered = paragraph.lower()
        return sum(lowered.count(term) for term in terms)

    best = max(paragraphs, key=score)
    if score(best) == 0:
        best = text.strip()
    if len(best) <= max_chars:
        return best
    return best[:max_chars].strip()
