"""Exploratory Data Analysis (EDA) for the German Credit dataset.

Run from the project root:
    python -m src.eda

Figures are saved to reports/figures/ and summary tables are printed.
"""

import logging

import matplotlib

matplotlib.use("Agg")  # save figures to files instead of opening windows

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_preprocessing import (
    NUMERICAL_FEATURES,
    PROJECT_ROOT,
    TARGET_COLUMN,
    load_raw_data,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
GOOD_COLOR = "#4C78A8"
BAD_COLOR = "#E45756"

# Human-readable names for the codes used in german.doc.
# Please cross-check these against data/raw/german.doc.
READABLE_LABELS = {
    "checking_account_status": {
        "A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM",
        "A14": "No checking account",
    },
    "credit_history": {
        "A30": "No credits / all paid", "A31": "All paid (this bank)",
        "A32": "Existing paid duly", "A33": "Delay in past",
        "A34": "Critical / other credits",
    },
    "savings_account": {
        "A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM",
        "A64": ">= 1000 DM", "A65": "Unknown / none",
    },
    "employment_since": {
        "A71": "Unemployed", "A72": "< 1 yr", "A73": "1-4 yrs",
        "A74": "4-7 yrs", "A75": ">= 7 yrs",
    },
    "housing": {"A151": "Rent", "A152": "Own", "A153": "Free"},
    "purpose": {
        "A40": "Car (new)", "A41": "Car (used)", "A42": "Furniture",
        "A43": "Radio/TV", "A44": "Appliances", "A45": "Repairs",
        "A46": "Education", "A48": "Retraining", "A49": "Business",
        "A410": "Other",
    },
}


def add_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add EDA-only columns. They must NEVER be used as model features."""
    df = df.copy()
    df["is_bad"] = (df[TARGET_COLUMN] == 2).astype(int)  # 1 = bad risk
    df["risk_label"] = df[TARGET_COLUMN].map({1: "Good", 2: "Bad"})
    return df


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save a figure into reports/figures/ and close it."""
    path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", path)


def bad_rate_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return applicant count and bad-risk rate (%) for each category."""
    grouped = df.groupby(column)["is_bad"].agg(applicants="count", bad_rate="mean")
    grouped["bad_rate_pct"] = (grouped["bad_rate"] * 100).round(1)
    grouped = grouped.drop(columns="bad_rate")
    labels = READABLE_LABELS.get(column, {})
    grouped.index = [labels.get(code, code) for code in grouped.index]
    grouped.index.name = column
    return grouped.sort_values("bad_rate_pct")


def plot_target_distribution(df: pd.DataFrame) -> None:
    counts = df["risk_label"].value_counts().reindex(["Good", "Bad"])
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(counts.index, counts.values, color=[GOOD_COLOR, BAD_COLOR])
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 10,
            f"{value} ({value / len(df):.0%})",
            ha="center",
        )
    ax.set_title("Target Distribution: Good vs Bad Credit Risk")
    ax.set_xlabel("Credit risk label")
    ax.set_ylabel("Number of applicants")
    ax.set_ylim(0, counts.max() * 1.15)
    save_figure(fig, "01_target_distribution.png")


def plot_numerical_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flat, NUMERICAL_FEATURES):
        ax.hist(df[col], bins=min(30, df[col].nunique()),
                color=GOOD_COLOR, edgecolor="white")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Number of applicants")
    for ax in axes.flat[len(NUMERICAL_FEATURES):]:
        ax.axis("off")  # hide the unused 8th panel
    save_figure(fig, "02_numerical_distributions.png")


def plot_numerical_by_risk(df: pd.DataFrame) -> None:
    columns = ["duration_months", "credit_amount", "age"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, col in zip(axes, columns):
        data = [df.loc[df["risk_label"] == "Good", col],
                df.loc[df["risk_label"] == "Bad", col]]
        box = ax.boxplot(data, patch_artist=True)
        for patch, color in zip(box["boxes"], [GOOD_COLOR, BAD_COLOR]):
            patch.set_facecolor(color)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Good", "Bad"])
        ax.set_title(f"{col} by credit risk")
        ax.set_xlabel("Credit risk label")
        ax.set_ylabel(col)
    save_figure(fig, "03_numerical_by_risk.png")


def plot_bad_rate_by_category(df: pd.DataFrame) -> None:
    overall = df["is_bad"].mean() * 100
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, col in zip(axes.flat, READABLE_LABELS.keys()):
        table = bad_rate_table(df, col)
        labels = [f"{name} (n={n})" for name, n in zip(table.index, table["applicants"])]
        ax.barh(labels, table["bad_rate_pct"].tolist(), color=BAD_COLOR)
        ax.axvline(overall, color="black", linestyle="--", linewidth=1,
                   label=f"Overall bad rate ({overall:.0f}%)")
        ax.set_title(f"Bad-risk rate by {col}")
        ax.set_xlabel("Bad-risk rate (%)")
        ax.set_ylabel(col)
        ax.legend(loc="lower right")
    save_figure(fig, "04_bad_rate_by_category.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df[NUMERICAL_FEATURES + ["is_bad"]].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, ax=ax)
    ax.set_title("Correlation Heatmap (numerical features and is_bad)")
    save_figure(fig, "05_correlation_heatmap.png")


def print_insights(df: pd.DataFrame) -> None:
    """Print the real numbers behind our business insights."""
    print(f"\n=== OVERALL BAD-RISK RATE: {df['is_bad'].mean() * 100:.1f}% ===")

    print("\n=== MEAN / MEDIAN BY RISK GROUP ===")
    summary = df.groupby("risk_label")[["duration_months", "credit_amount", "age"]]
    print(summary.agg(["mean", "median"]).round(1))

    for col in ["checking_account_status", "credit_history",
                "savings_account", "purpose"]:
        print(f"\n=== BAD-RISK RATE BY {col.upper()} ===")
        print(bad_rate_table(df, col))

    print("\n=== CORRELATION OF NUMERICAL FEATURES WITH is_bad ===")
    corr = df[NUMERICAL_FEATURES + ["is_bad"]].corr()["is_bad"]
    print(corr.drop("is_bad").sort_values(ascending=False).round(3))


def main() -> None:
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 30)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = add_helper_columns(load_raw_data())

    plot_target_distribution(df)
    plot_numerical_distributions(df)
    plot_numerical_by_risk(df)
    plot_bad_rate_by_category(df)
    plot_correlation_heatmap(df)
    print_insights(df)


if __name__ == "__main__":
    main()