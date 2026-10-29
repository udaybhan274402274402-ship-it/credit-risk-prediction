"""Compare candidate models with cross-validation on the TRAINING set only.

Run from the project root:
    python -m src.train

The test set is never touched here. It is reserved for the final evaluation.
"""

import logging
import time

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data_preprocessing import (
    PROJECT_ROOT,
    RANDOM_STATE,
    build_preprocessor,
    get_features_and_target,
    load_raw_data,
    split_data,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports"
CV_FOLDS = 5

# Positive class (1) = higher risk. zero_division=0 avoids warnings for the dummy model.
SCORERS = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def build_pipeline(model) -> Pipeline:
    """Preprocessing + model in ONE pipeline (so preprocessing is fitted per CV fold)."""
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", model)])


def get_candidate_models() -> dict:
    """Baseline plus the three required algorithms, with and without class weights."""
    rs = RANDOM_STATE
    return {
        "Dummy (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=rs),
        "Logistic Regression (balanced)": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=rs
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=rs),
        "Decision Tree (balanced)": DecisionTreeClassifier(
            class_weight="balanced", random_state=rs
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=rs, n_jobs=-1
        ),
        "Random Forest (balanced)": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=rs, n_jobs=-1
        ),
    }


def cross_validate_models(X_train, y_train) -> pd.DataFrame:
    """5-fold stratified CV for every candidate model (default settings)."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in get_candidate_models().items():
        start = time.time()
        scores = cross_validate(
            build_pipeline(model), X_train, y_train,
            cv=cv, scoring=SCORERS, return_train_score=True,
        )
        rows.append({
            "model": name,
            "accuracy": scores["test_accuracy"].mean(),
            "precision": scores["test_precision"].mean(),
            "recall": scores["test_recall"].mean(),
            "f1": scores["test_f1"].mean(),
            "roc_auc": scores["test_roc_auc"].mean(),
            "roc_auc_std": scores["test_roc_auc"].std(),
            "train_roc_auc": scores["train_roc_auc"].mean(),  # big gap vs roc_auc = overfitting
        })
        logger.info("Cross-validated %-32s (%.1fs)", name, time.time() - start)
    results = pd.DataFrame(rows).round(3)
    return results.sort_values("roc_auc", ascending=False).reset_index(drop=True)


def get_tuning_setup() -> dict:
    """Estimators and parameter grids. class_weight is part of the grid, so the DATA decides."""
    rs = RANDOM_STATE
    class_weights = [None, "balanced"]
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=rs),
            {"model__C": [0.01, 0.1, 1, 10], "model__class_weight": class_weights},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=rs),
            {
                "model__max_depth": [3, 4, 5, 6, 8],
                "model__min_samples_leaf": [5, 10, 20],
                "model__class_weight": class_weights,
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=rs),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_leaf": [1, 3, 5],
                "model__class_weight": class_weights,
            },
        ),
    }


def tune_models(X_train, y_train) -> pd.DataFrame:
    """Grid search with CV (scoring = ROC-AUC) for each algorithm."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, (estimator, grid) in get_tuning_setup().items():
        start = time.time()
        search = GridSearchCV(
            build_pipeline(estimator), param_grid=grid,
            scoring="roc_auc", cv=cv, n_jobs=-1,
        )
        search.fit(X_train, y_train)
        rows.append({
            "model": name,
            "best_cv_roc_auc": round(search.best_score_, 3),
            "best_params": str(search.best_params_),
        })
        logger.info("Tuned %-22s (%.1fs)", name, time.time() - start)
    return pd.DataFrame(rows).sort_values("best_cv_roc_auc", ascending=False)


def main() -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 120)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_data()
    X, y = get_features_and_target(df)
    X_train, _, y_train, _ = split_data(X, y)  # test part is deliberately ignored
    logger.info("Training set: %s | higher-risk rate: %.1f%%",
                X_train.shape, y_train.mean() * 100)

    cv_results = cross_validate_models(X_train, y_train)
    print("\n=== CROSS-VALIDATION RESULTS (default settings, train set only) ===")
    print(cv_results.to_string(index=False))
    cv_results.to_csv(REPORTS_DIR / "cv_model_comparison.csv", index=False)

    tuning_results = tune_models(X_train, y_train)
    print("\n=== HYPERPARAMETER TUNING (GridSearchCV, scoring = ROC-AUC) ===")
    print(tuning_results.to_string(index=False))
    tuning_results.to_csv(REPORTS_DIR / "tuning_results.csv", index=False)

    logger.info("Saved results to %s", REPORTS_DIR)


if __name__ == "__main__":
    main()