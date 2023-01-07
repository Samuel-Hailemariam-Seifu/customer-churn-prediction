from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config.settings import get_config
from src.data.loader import load_dataset, standardize_dataframe


def run_eda() -> None:
    cfg = get_config()
    df = standardize_dataframe(load_dataset())
    out_dir = Path("docs") / "eda_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    churn_counts = df[cfg.data.target_column].value_counts(dropna=False)
    plt.figure(figsize=(6, 4))
    churn_counts.plot(kind="bar", color=["#4CAF50", "#F44336"])
    plt.title("Churn Class Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution.png")
    plt.close()

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        corr = df[numeric_cols].corr(numeric_only=True)
        plt.figure(figsize=(8, 6))
        plt.imshow(corr, cmap="coolwarm", aspect="auto")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.colorbar()
        plt.title("Numerical Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(out_dir / "numerical_correlation.png")
        plt.close()

    for column in ["Contract", "InternetService", "PaymentMethod"]:
        if column in df.columns:
            pivot = (
                df.groupby(column)[cfg.data.target_column]
                .value_counts(normalize=True)
                .rename("ratio")
                .reset_index()
            )
            churn_ratio = pivot[pivot[cfg.data.target_column] == "Yes"].sort_values("ratio", ascending=False)
            plt.figure(figsize=(9, 4))
            plt.bar(churn_ratio[column], churn_ratio["ratio"])
            plt.title(f"Churn Ratio by {column}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(out_dir / f"churn_by_{column.lower()}.png")
            plt.close()

    summary_path = Path("docs") / "eda_summary.md"
    class_ratio = churn_counts / churn_counts.sum()
    lines = [
        "# EDA Summary",
        "",
        f"- Churn ratio: {class_ratio.get('Yes', 0):.2%}",
        f"- Non-churn ratio: {class_ratio.get('No', 0):.2%}",
        "- Month-to-month contracts generally show higher churn risk.",
        "- Fiber optic users and electronic check payments often have elevated churn in this dataset.",
        "- Tenure and total charge accumulation usually separate retained users from churners.",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
