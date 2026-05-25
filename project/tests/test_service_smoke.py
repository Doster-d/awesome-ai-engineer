from fastapi.testclient import TestClient

from contract_inspector.service.api import app
from contract_inspector.service.pipeline import ContractInspectionPipeline


def test_pipeline_returns_evidence_quote():
    text = "This Agreement shall be governed by the laws of New York. Liability is capped at fees paid."
    response = ContractInspectionPipeline().inspect(text, ["Governing Law"], top_k=2)

    result = response["results"][0]
    assert result["found"] is True
    assert result["evidence"]
    assert result["evidence"][0]["quote"] in text


def test_api_health_and_predict():
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    predict = client.post(
        "/predict",
        json={
            "contract_text": "This Agreement shall be governed by the laws of New York.",
            "clause_types": ["Governing Law"],
            "top_k": 1,
            "use_rlm": False,
        },
    )
    assert predict.status_code == 200
    assert predict.json()["results"][0]["found"] is True


def test_pipeline_rlm_falls_back_without_openrouter_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    text = "This Agreement shall be governed by the laws of New York."

    response = ContractInspectionPipeline().inspect(text, ["Governing Law"], top_k=1, use_rlm=True)

    result = response["results"][0]
    assert response["model_version"].endswith("_rlm_fallback")
    assert result["found"] is True
    assert result["rlm_trace"]
