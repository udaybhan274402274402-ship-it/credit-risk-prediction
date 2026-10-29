"""Final evaluation on the held-out TEST set.

Run from the project root:
    python -m src.evaluate

Rules decided BEFORE looking at the test set:
  * Final model = highest cross-validated ROC-AUC from tuning (Random Forest).
  * Tuned Logistic Regression is evaluated alongside as an interpretable reference.
  * The decision threshold is chosen ONLY from out-of-fold predictions on the
    TRAINING set. The test set is used once, for reporting.
"""

import logging

import matplotlib

matplotlib.use("Agg")  # save figures to files instead of opening windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.data_preprocessing import (
    PROJECT_ROOT,
    RANDOM_STATE,
    get_features_and_target,
    load_raw_data,
    split_data,
)
from src.train import CV_FOLDS, build_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Cost matrix from german.doc: approving a bad applicant (false negative) costs 5,
# rejecting a good applicant (false positive) costs 1.
# Please verify these numbers in data/raw/german.doc.
COST_FN = 5
COST_FP = 1

FINAL_MODEL = "Random Forest (tuned)"

# Best parameters taken from the GridSearchCV output of src/train.py.
BEST_MODELS = {
    "Random Forest (tuned)": RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=3,
        class_weight=None, random_state=RANDOM_STATE, n_jobs=-1,
    ),
    "Logistic Regression (tuned)": LogisticRegression(
        C=0.1, class_weight=None, max_iter=1000, random_state=RANDOM_STATE,
    ),
}

RESULT_COLUMNS = [
    "model", "setting", "threshold", "accuracy", "precision", "recall", "f1",
    "roc_auc", "brier", "tn", "fp", "fn", "tp", "total_cost", "cost_per_applicant",
]


def save_figure(fig: plt.Figure, filename: str) -> None:
    path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", path)


def total_cost(y_true, y_pred) -> int:
    """Total cost of the errors: false negatives and false positives."""
    _, fp, fn, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(fp * COST_FP + fn * COST_FN)


def choose_threshold(y_true, proba):
    """Pick the threshold with the lowest total cost (on training out-of-fold data)."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    costs = np.array([total_cost(y_true, (proba >= t).astype(int)) for t in grid])
    return float(grid[int(np.argmin(costs))]), grid, costs


def compute_metrics(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    cost = int(fp * COST_FP + fn * COST_FN)
    return {
        "threshold": round(threshold, 2),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),  # threshold independent
        "brier": brier_score_loss(y_true, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "total_cost": cost,
        "cost_per_applicant": cost / len(y_true),
    }


def naive_baseline_rows(y_true) -> list:
    """Cost of the two trivial strategies: approve everyone / reject everyone."""
    n_bad = int(y_true.sum())
    n_good = len(y_true) - n_bad
    rows = []
    for name, cost in [("Naive: approve everyone", n_bad * COST_FN),
                       ("Naive: reject everyone", n_good * COST_FP)]:
        rows.append({"model": name, "setting": "-", "total_cost": cost,
                     "cost_per_applicant": cost / len(y_true)})
    return rows


def plot_confusion_matrices(y_true, proba, threshold: float, name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    settings = [("default threshold", 0.5), ("cost-based threshold", threshold)]
    for ax, (label, t) in zip(axes, settings):
        cm = confusion_matrix(y_true, (proba >= t).astype(int), labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Lower risk", "Higher risk"],
                    yticklabels=["Lower risk", "Higher risk"])
        ax.set_title(f"{name}: {label} ({t:.2f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    save_figure(fig, "06_confusion_matrices.png")


def plot_roc_curves(y_true, test_proba: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, proba in test_proba.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_true, proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random guess")
    ax.set_title("ROC Curves on the Test Set")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.legend(loc="lower right")
    save_figure(fig, "07_roc_curves.png")


def plot_calibration_curves(y_true, test_proba: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, proba in test_proba.items():
        frac_positive, mean_predicted = calibration_curve(
            y_true, proba, n_bins=5, strategy="quantile"
        )
        ax.plot(mean_predicted, frac_positive, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_title("Calibration Curves on the Test Set (only 200 applicants, noisy)")
    ax.set_xlabel("Mean predicted probability of higher risk")
    ax.set_ylabel("Observed share of higher-risk applicants")
    ax.legend(loc="upper left")
    save_figure(fig, "08_calibration_curves.png")


def plot_threshold_cost(cost_curves: dict, chosen: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (grid, costs) in cost_curves.items():
        line, = ax.plot(grid, costs, label=name)
        ax.axvline(chosen[name], color=line.get_color(), linestyle=":")
    ax.set_title("Total Cost vs Threshold (training out-of-fold predictions)")
    ax.set_xlabel("Decision threshold for 'higher risk'")
    ax.set_ylabel(f"Total cost (FN = {COST_FN}, FP = {COST_FP})")
    ax.legend()
    save_figure(fig, "09_threshold_cost_curve.png")


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_data()
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows, reports = [], []
    test_proba, chosen, cost_curves = {}, {}, {}

    for name, model in BEST_MODELS.items():
        pipeline = build_pipeline(model)

        # Step A: choose the threshold from TRAINING out-of-fold probabilities only.
        oof_proba = cross_val_predict(
            pipeline, X_train, y_train, cv=cv, method="predict_proba"
        )[:, 1]
        threshold, grid, costs = choose_threshold(y_train.to_numpy(), oof_proba)
        chosen[name] = threshold
        cost_curves[name] = (grid, costs)

        # Step B: fit on the full training set, then touch the test set once.
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        test_proba[name] = proba

        for setting, t in [("default 0.50", 0.5), ("cost-based", threshold)]:
            row = {"model": name, "setting": setting}
            row.update(compute_metrics(y_test, proba, t))
            rows.append(row)
            report = classification_report(
                y_test, (proba >= t).astype(int),
                target_names=["Lower risk (0)", "Higher risk (1)"], zero_division=0,
            )
            reports.append(
                f"=== {name} | {setting} (threshold {t:.2f}) ===\n{report}"
            )
        logger.info("%s: cost-based threshold = %.2f", name, threshold)

    rows.extend(naive_baseline_rows(y_test))
    results = pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)
    results.to_csv(REPORTS_DIR / "evaluation_results.csv", index=False)
    (REPORTS_DIR / "classification_reports.txt").write_text(
        "\n\n".join(reports), encoding="utf-8"
    )

    print(f"\n=== TEST SET RESULTS (n = {len(y_test)}; cost: FN = {COST_FN}, FP = {COST_FP}) ===")
    print(results.round(3).to_string(index=False))

    for report in reports:
        if report.startswith(f"=== {FINAL_MODEL}"):
            print("\n" + report)

    plot_confusion_matrices(y_test, test_proba[FINAL_MODEL], chosen[FINAL_MODEL], FINAL_MODEL)
    plot_roc_curves(y_test, test_proba)
    plot_calibration_curves(y_test, test_proba)
    plot_threshold_cost(cost_curves, chosen)

    logger.info("Saved evaluation_results.csv and classification_reports.txt in %s", REPORTS_DIR)


if __name__ == "__main__":
    main()