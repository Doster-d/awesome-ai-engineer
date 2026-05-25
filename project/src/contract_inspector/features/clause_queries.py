from contract_inspector.settings.domain_config import load_domain_terms


def clause_queries() -> dict[str, list[str]]:
    data = load_domain_terms()
    return {str(key): [str(item) for item in value] for key, value in data.get("clause_queries", {}).items()}


def build_clause_query(clause_type: str) -> str:
    terms = clause_queries().get(clause_type, [clause_type])
    return " ".join(terms)
