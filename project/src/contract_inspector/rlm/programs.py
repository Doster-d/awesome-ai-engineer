import json
import os

import dspy

from contract_inspector.rlm.signatures import JudgeEvidence, StructureClauseAnswer


class ClauseEvidenceProgram(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.judge = dspy.ChainOfThought(JudgeEvidence)
        self.structure = dspy.ChainOfThought(StructureClauseAnswer)

    def forward(self, clause_type: str, evidence_quotes: str) -> dspy.Prediction:
        judge_outputs = []
        quotes = _parse_quotes(evidence_quotes)
        max_quotes = int(os.getenv("RLM_MAX_QUOTES", "10"))
        for quote in quotes[:max_quotes]:
            judged = self.judge(
                clause_type=clause_type,
                candidate_quote=quote["quote"],
                surrounding_context=quote.get("context", quote["quote"]),
            )
            judge_outputs.append(
                {
                    "chunk_id": quote.get("chunk_id"),
                    "quote": quote["quote"],
                    "verdict": judged.verdict,
                    "rationale": judged.rationale,
                }
            )

        supported = [item for item in judge_outputs if "support" in item["verdict"].lower()]
        structure_input = json.dumps(supported or judge_outputs, ensure_ascii=False)
        structured = self.structure(clause_type=clause_type, evidence_quotes=structure_input)
        return dspy.Prediction(answer_json=structured.answer_json, judge_outputs=json.dumps(judge_outputs, ensure_ascii=False))


def _parse_quotes(evidence_quotes: str) -> list[dict]:
    try:
        data = json.loads(evidence_quotes)
    except json.JSONDecodeError:
        return [{"quote": evidence_quotes}]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict) and item.get("quote")]
    return [{"quote": evidence_quotes}]
