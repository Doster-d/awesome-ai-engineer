import dspy


class PlanClauseSearch(dspy.Signature):
    clause_type: str = dspy.InputField()
    contract_outline: str = dspy.InputField()
    search_plan_json: str = dspy.OutputField(
        desc="JSON with search queries, grep patterns, likely sections, stopping criteria"
    )


class JudgeEvidence(dspy.Signature):
    clause_type: str = dspy.InputField()
    candidate_quote: str = dspy.InputField()
    surrounding_context: str = dspy.InputField()
    verdict: str = dspy.OutputField(desc="supports, weak_support, unrelated")
    rationale: str = dspy.OutputField(desc="Reason in the same language as clause_type")


class StructureClauseAnswer(dspy.Signature):
    clause_type: str = dspy.InputField()
    evidence_quotes: str = dspy.InputField()
    answer_json: str = dspy.OutputField(
        desc="Valid JSON with found, answer, risk_level, confidence, evidence. Use the same language as clause_type"
    )


class VerifyContractHypothesis(dspy.Signature):
    hypothesis: str = dspy.InputField()
    evidence_quotes: str = dspy.InputField()
    label: str = dspy.OutputField(desc="entailment, contradiction, not_mentioned")
    rationale: str = dspy.OutputField(desc="Reason in the same language as hypothesis")
