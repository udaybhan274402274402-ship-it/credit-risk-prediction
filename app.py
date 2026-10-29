"""Streamlit dashboard: Credit Risk Prediction System (educational project).

Run from the project root:
    streamlit run app.py
"""

import logging

import pandas as pd
import streamlit as st

from src.data_preprocessing import (
    MODEL_CATEGORICAL_FEATURES,
    MODEL_NUMERICAL_FEATURES,
    NUMERICAL_FEATURES,
    PROJECT_ROOT,
    TARGET_COLUMN,
    load_raw_data,
)
from src.labels import FEATURE_TITLES, describe_value
from src.predict import HARD_BOUNDS, ValidationError, load_artifacts, predict_applicant

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Credit Risk Prediction (Educational)", page_icon="📊", layout="wide")

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# label, default value (about the training median), step
NUMERIC_INPUTS = {
    "duration_months": ("Loan duration (months)", 18, 1),
    "credit_amount": ("Credit amount (DM)", 2300, 100),
    "installment_rate": ("Installment rate (% of disposable income, 1 to 4)", 3, 1),
    "residence_since": ("Present residence since (1 to 4)", 3, 1),
    "age": ("Age (years)", 33, 1),
    "existing_credits": ("Existing credits at this bank", 1, 1),
    "num_dependents": ("Number of dependents", 1, 1),
}


@st.cache_resource
def get_artifacts():
    """Load the saved pipeline and metadata once."""
    return load_artifacts()


@st.cache_data
def get_dataset() -> pd.DataFrame:
    return load_raw_data()


def show_figure(filename: str, caption: str) -> None:
    path = FIGURES_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"Figure {filename} not found yet. Run the matching script first.")


def show_table(filename: str, title: str) -> None:
    path = REPORTS_DIR / filename
    st.markdown(f"**{title}**")
    if path.exists():
        st.dataframe(pd.read_csv(path).round(3), hide_index=True)
    else:
        st.info(f"{filename} not found yet.")


try:
    pipeline, metadata = get_artifacts()
except FileNotFoundError:
    pipeline, metadata = None, None

st.title("📊 Credit Risk Prediction System")
st.warning(
    "**Educational decision-support project.** This is NOT a real banking system. "
    "Its output must never be the sole basis for an actual lending decision."
)

tab_overview, tab_data, tab_perf, tab_eda, tab_predict, tab_limits = st.tabs(
    ["Overview", "Dataset", "Model performance", "EDA", "Try a prediction", "Limitations & ethics"]
)

# ---------------------------------------------------------------- Overview
with tab_overview:
    st.header("Project overview")
    st.markdown(
        """
This project estimates whether a loan applicant looks like a **higher-risk** or **lower-risk**
case, using the public German Credit dataset and machine learning.

**How it works**
1. The dataset is split into training and test parts (stratified, fixed random seed).
2. A scikit-learn Pipeline does preprocessing (scaling and one-hot encoding) and the model together,
   fitted on the training data only (no data leakage).
3. Logistic Regression, Decision Tree and Random Forest were compared with cross-validation.
4. The final model was evaluated once on the untouched test set.
5. This app loads the *same* saved pipeline that was trained and evaluated.
        """
    )
    if metadata:
        st.caption(
            f"Final model: **{metadata['model_name']}**. Created: {metadata['created_utc']} (UTC)."
        )

# ----------------------------------------------------------------- Dataset
with tab_data:
    st.header("Dataset summary")
    st.markdown(
        "**Statlog (German Credit Data)**, UCI Machine Learning Repository "
        "(Hofmann, 1994, DOI 10.24432/C5NC77), licensed CC BY 4.0."
    )
    try:
        data = get_dataset()
        higher_risk_share = (data[TARGET_COLUMN] == 2).mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Applicants", f"{len(data):,}")
        c2.metric("Input columns in raw data", data.shape[1] - 1)
        c3.metric("Labelled higher risk", f"{higher_risk_share:.0%}")
        st.markdown("**Numerical features (summary statistics)**")
        st.dataframe(data[NUMERICAL_FEATURES].describe().T.round(2))
    except FileNotFoundError:
        st.info("Raw data not found. Run `python -m src.download_data` to see the dataset summary.")
    st.markdown(
        "The target is a **good/bad credit risk label**. The dataset description does not "
        "define it as a verified default event, so treat it as a risk label, not proven default."
    )

# ------------------------------------------------------ Model performance
with tab_perf:
    st.header("Model performance")
    st.markdown(
        "Models were compared with 5-fold cross-validation on the training set. "
        "The test set (200 applicants) was used once for the final check."
    )
    show_table("cv_model_comparison.csv", "Cross-validation comparison (default settings)")
    show_table("tuning_results.csv", "Hyperparameter tuning (scoring = ROC-AUC)")
    show_table("evaluation_results.csv", "Final test-set results (default 0.50 vs cost-based threshold)")
    st.caption(
        "Accuracy alone is misleading here: a model that always says 'lower risk' already "
        "scores 70%. Compare recall, precision and total cost as well."
    )
    left, right = st.columns(2)
    with left:
        show_figure("06_confusion_matrices.png", "Confusion matrices at two thresholds")
        show_figure("08_calibration_curves.png", "Calibration curves (only 200 test applicants, noisy)")
    with right:
        show_figure("07_roc_curves.png", "ROC curves on the test set")
        show_figure("09_threshold_cost_curve.png", "Total cost vs decision threshold")

# --------------------------------------------------------------------- EDA
with tab_eda:
    st.header("Exploratory data analysis")
    show_figure("01_target_distribution.png", "How many applicants are labelled good vs bad.")
    show_figure("02_numerical_distributions.png", "How the numerical features are spread out.")
    show_figure("03_numerical_by_risk.png", "Loan duration, credit amount and age for good vs bad groups.")
    show_figure("04_bad_rate_by_category.png", "Share of higher-risk labels inside each category (n = group size).")
    show_figure("05_correlation_heatmap.png", "Linear relationships between numerical features and the risk label.")
    show_figure("10_permutation_importance.png", "Which features the model relies on (not proof of cause and effect).")

# ------------------------------------------------------------------ Predict
with tab_predict:
    st.header("Try a prediction")
    if pipeline is None:
        st.error("Model files not found. Run `python -m src.save_model` first, then reload this page.")
    else:
        st.info(
            "Enter fictional applicant details. Nothing you type is saved. "
            "Sensitive attributes (gender/marital status, foreign-worker status) are not model inputs."
        )
        with st.form("applicant_form"):
            left, right = st.columns(2)
            applicant = {}
            with left:
                st.subheader("Loan and applicant numbers")
                for col in MODEL_NUMERICAL_FEATURES:
                    label, default, step = NUMERIC_INPUTS[col]
                    low, high = HARD_BOUNDS[col]
                    applicant[col] = st.number_input(
                        label, min_value=low, max_value=high, value=default, step=step
                    )
            with right:
                st.subheader("Applicant background")
                for col in MODEL_CATEGORICAL_FEATURES:
                    options = metadata["categorical_features"][col]
                    applicant[col] = st.selectbox(
                        FEATURE_TITLES.get(col, col),
                        options,
                        format_func=lambda code, c=col: describe_value(c, code),
                    )
            submitted = st.form_submit_button("Estimate risk")

        if submitted:
            try:
                result = predict_applicant(applicant, pipeline, metadata)
            except ValidationError as error:
                st.error(f"Invalid input: {error}")
            except Exception:
                logger.exception("Prediction failed")
                st.error("Something went wrong while predicting. Please check the inputs and try again.")
            else:
                for message in result.warnings:
                    st.warning(message)
                if result.higher_risk:
                    st.error("Model flag: **HIGHER risk**")
                else:
                    st.success("Model flag: **LOWER risk**")
                m1, m2 = st.columns(2)
                m1.metric("Risk score (0 to 1)", f"{result.risk_score:.2f}")
                m2.metric("Decision threshold", f"{result.threshold:.2f}")
                st.caption(
                    "The risk score is a model output used for ranking. It has not been verified as a "
                    "true probability of default, so do not read 0.40 as 'a 40% chance of default'."
                )
                with st.expander("How to read this result"):
                    st.markdown(
                        f"""
- The applicant is flagged as higher risk when the score is **{result.threshold:.2f} or more**.
- That threshold is deliberately low. It was chosen on training data because wrongly approving a
  risky applicant was assumed to cost much more than wrongly rejecting a good one.
- A low threshold means **many lower-risk applicants are also flagged**.
- The model learned patterns from 1994 German data. It cannot explain *why* someone is risky.
                        """
                    )

# ------------------------------------------------------------- Limitations
with tab_limits:
    st.header("Limitations and ethical considerations")
    if metadata:
        t = metadata["test_metrics"]
        st.markdown(
            f"**On the 200-applicant test set** (cost-based threshold {t['threshold']:.2f}): "
            f"recall {t['recall']:.0%}, precision {t['precision']:.0%}, "
            f"{int(t['fp'])} lower-risk applicants wrongly flagged, {int(t['fn'])} risky applicants missed."
        )
    st.markdown(
        """
- **Educational only.** Not validated, audited or approved for real lending.
- **Old, small dataset.** 1,000 applicants from 1994. Today's applicants and economy differ.
- **The label is a risk label,** not a confirmed default record.
- **Many false alarms by design.** The threshold favours catching risky applicants, so precision is low.
- **Chance and uncertainty.** The test set is small, so small differences between models are noise.
- **Feature importance is not causation.** It shows what the model uses, not why people default.
- **Fairness.** Gender/marital status and foreign-worker status are excluded from the model, but
  other features can act as proxies. Age is still used. Group-level checks were done on very small
  groups and prove neither fairness nor bias.
- **Privacy.** Real applicant data is sensitive. This app stores nothing and uses a public dataset.
        """
    )