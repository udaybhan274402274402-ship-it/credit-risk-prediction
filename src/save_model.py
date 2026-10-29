"""Train the FINAL pipeline on the training split and save it with its metadata.

Run from the project root (after python -m src.evaluate):
    python -m src.save_model

The saved model is exactly the one evaluated on the test set, so the reported
test metrics describe this artifact.
"""

import json
import logging
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from src.data_preprocessing import (
    MODEL_CATEGORICAL_FEATURES,
    MODEL_NUMERICAL_FEATURES,
    PROJECT_ROOT,
    RANDOM_STATE,
    TEST_SIZE,
    get_features_and_target,
    load_raw_data,
    split_data,
)
from src.evaluate import BEST_MODELS, FINAL_MODEL
from src.explain import load_chosen_threshold
from src.train import build_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODELS_DIR / "credit_risk_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


def read_test_metrics() -> dict:
    """Read the test metrics that src/evaluate.py saved for the final model."""
    results = pd.read_csv(REPORTS_DIR / "evaluation_results.csv")
    row = results[(results["model"] == FINAL_MODEL) & (results["setting"] == "cost-based")]
    if row.empty:
        raise ValueError("Evaluation results not found. Run: python -m src.evaluate")
    keep = ["threshold", "accuracy", "precision", "recall", "f1", "roc_auc",
            "brier", "tn", "fp", "fn", "tp", "total_cost"]
    return {key: float(row[key].iloc[0]) for key in keep}


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_data()
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    pipeline = build_pipeline(clone(BEST_MODELS[FINAL_MODEL]))
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODEL_PATH)
    logger.info("Saved pipeline: %s", MODEL_PATH)

    metadata = {
        "model_name": FINAL_MODEL,
        "positive_class": "higher risk (1)",
        "threshold": load_chosen_threshold(),
        "threshold_note": "Chosen on training out-of-fold predictions with cost FN=5, FP=1.",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "numerical_features": {
            col: {"min": float(X_train[col].min()), "max": float(X_train[col].max())}
            for col in MODEL_NUMERICAL_FEATURES
        },
        "categorical_features": {
            col: sorted(str(v) for v in X_train[col].unique())
            for col in MODEL_CATEGORICAL_FEATURES
        },
        "test_metrics": read_test_metrics(),
        "dataset": "Statlog (German Credit Data), UCI ML Repository, CC BY 4.0, "
                   "doi:10.24432/C5NC77",
        "disclaimer": "Educational decision-support project. Not for real lending decisions.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved metadata: %s", METADATA_PATH)

    # Sanity check: the reloaded file must behave exactly like the in-memory model.
    reloaded = joblib.load(MODEL_PATH)
    proba_saved = reloaded.predict_proba(X_test)[:, 1]
    proba_memory = pipeline.predict_proba(X_test)[:, 1]
    print("\n=== SANITY CHECK ===")
    print("Reloaded model matches in-memory model:", bool(np.allclose(proba_saved, proba_memory)))
    print("Reloaded model test ROC-AUC:", round(roc_auc_score(y_test, proba_saved), 3))
    print("Threshold stored in metadata:", metadata["threshold"])


if __name__ == "__main__":
    main()