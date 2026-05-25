from contract_inspector.features.clause_queries import build_clause_query


def test_known_clause_query_uses_synonyms():
    query = build_clause_query("Cap On Liability")

    assert "limitation of liability" in query
    assert "maximum liability" in query


def test_unknown_clause_query_falls_back_to_clause_type():
    assert build_clause_query("Custom Clause") == "Custom Clause"
