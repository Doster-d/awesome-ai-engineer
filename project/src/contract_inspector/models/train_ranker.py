import json
import hashlib

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from contract_inspector.features.retrieval_features import build_baseline_candidates
from contract_inspector.models.metrics import ranking_metrics
from contract_inspector.retrieval.schemas import RetrievalHit
from contract_inspector.settings.paths import PROJECT_DIR


def main() -> None:
    artifacts_dir = PROJECT_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = artifacts_dir / "baseline_candidates.parquet"
    if candidates_path.exists():
        frame = pd.read_parquet(candidates_path)
    else:
        frame = build_baseline_candidates()
        frame.to_parquet(candidates_path, index=False)

    train_mask = frame["contract_id"].map(_stable_bucket) < 80
    train = frame[train_mask].copy()
    test = frame[~train_mask].copy()

    numeric_features = [
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
    ]
    categorical_features = ["clause_type"]
    model = Pipeline(
        steps=[
            (
                "features",
                ColumnTransformer(
                    transformers=[
                        ("numeric", StandardScaler(), numeric_features),
                        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                    ]
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(train[numeric_features + categorical_features], train["is_relevant"].astype(int))
    test_scores = model.predict_proba(test[numeric_features + categorical_features])[:, 1]
    test = test.assign(ranker_score=test_scores)
    metrics = {
        "model": "logistic_regression",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "positive_rate_train": float(train["is_relevant"].mean()),
        "positive_rate_test": float(test["is_relevant"].mean()),
        "roc_auc": float(roc_auc_score(test["is_relevant"].astype(int), test_scores)),
        "average_precision": float(average_precision_score(test["is_relevant"].astype(int), test_scores)),
        "ranking": _evaluate_ranked_frame(test),
    }
    joblib.dump(model, artifacts_dir / "feature_ranker.joblib")
    (artifacts_dir / "ranker_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def _stable_bucket(contract_id: str) -> int:
    digest = hashlib.sha1(contract_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def _evaluate_ranked_frame(frame: pd.DataFrame) -> dict:
    totals = {"recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0}
    groups = 0
    for _, group in frame.groupby("example_id"):
        relevant_ids = set(group.loc[group["is_relevant"], "chunk_id"])
        if not relevant_ids:
            continue
        ranked = group.sort_values("ranker_score", ascending=False)
        hits = [
            RetrievalHit(
                query=row["query"],
                chunk_id=row["chunk_id"],
                contract_id=row["contract_id"],
                score=float(row["ranker_score"]),
                rank=rank,
                text="",
                char_start=0,
                char_end=0,
            )
            for rank, (_, row) in enumerate(ranked.iterrows(), start=1)
        ]
        group_metrics = ranking_metrics(hits, relevant_ids)
        for key in totals:
            totals[key] += group_metrics[key]
        groups += 1
    return {key: value / groups if groups else 0.0 for key, value in totals.items()} | {"evaluated": groups}


if __name__ == "__main__":
    main()
