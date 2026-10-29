"""Explainability and fairness audit for the final model.

Run from the project root (after python -m src.evaluate):
    python -m src.explain

Notes:
  * Sensitive attributes (personal_status_sex, foreign_worker) are NOT model inputs.
    They are used here only to audit how the model behaves across groups.
  * Feature importance describes what the model uses. It is NOT causal evidence.
"""

import logging

import matplotlib

matplotlib.use("Agg")  # save figures to files instead of opening windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix

from src.data_preprocessing import (
    PROJECT_ROOT,
    RANDOM_STATE,
    get_features_and_target,
    load_raw_data,
    split_data,
)
from src.evaluate import BEST_MODELS, FINAL_MODEL
from src.train import build_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

LR_NAME = "Logistic Regression (tuned)"
MIN_GROUP_SIZE = 30  # groups smaller than this are too small to judge

AGE_BINS = [0, 25, 40, 200]
AGE_LABELS = ["<=25", "26-40", ">40"]

# Please cross-check these labels with data/raw/german.doc (attributes 9 and 20).
SEX_STATUS_LABELS = {
    "A91": "A91 male: divorced/separated",
    "A92": "A92 female: divorced/separated/married",
    "A93": "A93 male: single",
    "A94": "A94 male: married/widowed",
}
FOREIGN_LABELS = {"A201": "A201 foreign worker: yes", "A202": "A202 foreign worker: no"}


def load_chosen_threshold() -> float:
    """Read the cost-based threshold that src/evaluate.py chose from TRAINING data."""
    results = pd.read_csv(REPORTS_DIR / "evaluation_results.csv")
    row = results[(results["model"] == FINAL_MODEL) & (results["setting"] == "cost-based")]
    if row.empty:
        raise ValueError("No cost-based row found. Run: python -m src.evaluate")
    return float(row["threshold"].iloc[0])


def compute_permutation_importance(pipeline, X_test, y_test) -> pd.DataFrame:
    """Drop in test ROC-AUC when one ORIGINAL feature is shuffled (30 repeats)."""
    result = permutation_importance(
        pipeline, X_test, y_test, scoring="roc_auc",
        n_repeats=30, random_state=RANDOM_STATE, n_jobs=-1,
    )
    table = pd.DataFrame({
        "feature": X_test.columns,
        "auc_drop_mean": result.importances_mean,
        "auc_drop_std": result.importances_std,
    })
    return table.sort_values("auc_drop_mean", ascending=False).reset_index(drop=True)


def plot_importance(table: pd.DataFrame) -> None:
    top = table.head(10).iloc[::-1]  # reverse so the biggest bar is on top
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top["feature"], top["auc_drop_mean"], xerr=top["auc_drop_std"],
            color="#4C78A8")
    ax.set_title("Permutation Importance on the Test Set (Random Forest)")
    ax.set_xlabel("Mean drop in ROC-AUC (error bar = std over 30 shuffles)")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    path = FIGURES_DIR / "10_permutation_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", path)


def logistic_coefficients(X_train, y_train) -> pd.DataFrame:
    """Fit the tuned Logistic Regression and return coefficients and odds ratios."""
    pipeline = build_pipeline(clone(BEST_MODELS[LR_NAME]))
    pipeline.fit(X_train, y_train)
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_[0]
    table = pd.DataFrame({
        "feature": names,
        "coefficient": coefs,
        "odds_ratio": np.exp(coefs),
    })
    return table.sort_values("coefficient", ascending=False).reset_index(drop=True)


def fairness_audit(df, X_test, y_test, proba, threshold: float) -> pd.DataFrame:
    """Compare flagging and error rates across groups (descriptive, not a verdict)."""
    original = df.loc[X_test.index]
    frame = pd.DataFrame({
        "y_true": y_test.to_numpy(),
        "y_pred": (proba >= threshold).astype(int),
        "age_group": pd.cut(X_test["age"], bins=AGE_BINS, labels=AGE_LABELS)
        .astype(str).to_numpy(),
        "personal_status_sex": original["personal_status_sex"]
        .replace(SEX_STATUS_LABELS).to_numpy(),
        "foreign_worker": original["foreign_worker"]
        .replace(FOREIGN_LABELS).to_numpy(),
    })

    rows = []
    for column in ["age_group", "personal_status_sex", "foreign_worker"]:
        for group, part in frame.groupby(column):
            y_true = part["y_true"].to_numpy()
            y_pred = part["y_pred"].to_numpy()
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            rows.append({
                "attribute": column,
                "group": group,
                "n": len(part),
                "n_bad": int(y_true.sum()),
                "actual_bad_rate": y_true.mean(),
                "flagged_rate": y_pred.mean(),
                "recall": tp / (tp + fn) if (tp + fn) else np.nan,
                "false_positive_rate": fp / (fp + tn) if (fp + tn) else np.nan,
                "enough_data": len(part) >= MIN_GROUP_SIZE,
            })
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 60)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_data()
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    threshold = load_chosen_threshold()

    final_pipeline = build_pipeline(clone(BEST_MODELS[FINAL_MODEL]))
    final_pipeline.fit(X_train, y_train)

    # 1. Permutation importance (original features, final model, test set)
    importance = compute_permutation_importance(final_pipeline, X_test, y_test)
    print("\n=== PERMUTATION IMPORTANCE (final model, test set, top 10) ===")
    print(importance.head(10).round(4).to_string(index=False))
    importance.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)
    plot_importance(importance)

    # 2. Logistic Regression coefficients (interpretable reference model)
    coefs = logistic_coefficients(X_train, y_train)
    print("\n=== LOGISTIC REGRESSION: strongest RISK-INCREASING terms ===")
    print(coefs.head(8).round(3).to_string(index=False))
    print("\n=== LOGISTIC REGRESSION: strongest RISK-DECREASING terms ===")
    print(coefs.tail(8).iloc[::-1].round(3).to_string(index=False))
    coefs.to_csv(REPORTS_DIR / "logistic_coefficients.csv", index=False)

    # 3. Fairness audit at the cost-based threshold
    proba = final_pipeline.predict_proba(X_test)[:, 1]
    audit = fairness_audit(df, X_test, y_test, proba, threshold)
    print(f"\n=== FAIRNESS AUDIT (threshold {threshold:.2f}; groups below "
          f"{MIN_GROUP_SIZE} applicants are unreliable) ===")
    print(audit.round(2).to_string(index=False))
    audit.to_csv(REPORTS_DIR / "fairness_audit.csv", index=False)

    logger.info("Saved explainability and fairness files in %s", REPORTS_DIR)


if __name__ == "__main__":
    main()