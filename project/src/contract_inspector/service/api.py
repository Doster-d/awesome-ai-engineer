from fastapi import FastAPI

from contract_inspector.service.pipeline import ContractInspectionPipeline
from contract_inspector.service.schemas import PredictRequest, PredictResponse


app = FastAPI(title="Contract Inspector RLM", version="0.1.0")
pipeline = ContractInspectionPipeline()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_version": pipeline.model_version,
        "artifacts_loaded": True,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> dict:
    return pipeline.inspect(
        contract_text=request.contract_text,
        clause_types=request.clause_types,
        top_k=request.top_k,
        use_rlm=request.use_rlm,
    )


def main() -> None:
    import uvicorn

    uvicorn.run("contract_inspector.service.api:app", host="0.0.0.0", port=8000)


def main_dev() -> None:
    import uvicorn

    uvicorn.run("contract_inspector.service.api:app", host="0.0.0.0", port=8000, reload=True)
