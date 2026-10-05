"""Generate plots and text reports from fraud benchmark results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def _save_pr_auc_chart(dataset: str, rows: pd.DataFrame, output_dir: Path) -> None:
    ordered = rows.sort_values("pr_auc", ascending=True)
    fig, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    bars = axis.barh(ordered["model"], ordered["pr_auc"], color=sns.color_palette("crest", len(ordered)))
    axis.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    axis.set_xlim(0, min(1.0, max(0.1, float(ordered["pr_auc"].max()) * 1.18)))
    axis.set_xlabel("Average precision (PR-AUC)")
    axis.set_title(f"{dataset.replace('_', ' ').title()} model comparison")
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    fig.savefig(output_dir / f"{dataset}_pr_auc.png", dpi=180)
    plt.close(fig)


def _save_confusion_matrix(
    dataset: str,
    model_name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    pr_auc: float,
    output_dir: Path,
) -> None:
    matrix = confusion_matrix(y_true, scores >= 0.5, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(6.5, 5.2), constrained_layout=True)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ax=axis,
    )
    axis.set_xlabel("Predicted label (threshold = 0.5)")
    axis.set_ylabel("Actual label")
    axis.set_title(f"Best model: {model_name} | PR-AUC = {pr_auc:.4f}")
    fig.savefig(output_dir / f"{dataset}_best_cm.png", dpi=180)
    plt.close(fig)


def _save_probability_curves(
    dataset: str,
    rows: pd.DataFrame,
    y_true: np.ndarray,
    scores_by_model: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    palette = sns.color_palette("colorblind", len(scores_by_model))
    for color, (_, row) in zip(palette, rows.iterrows()):
        model_name = str(row["model"])
        scores = scores_by_model[model_name]
        precision, recall, _ = precision_recall_curve(y_true, scores)
        axes[0].plot(recall, precision, label=f"{model_name} (AP={row['pr_auc']:.3f})", color=color)
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, scores)
        axes[1].plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{model_name} (AUC={row['roc_auc']:.3f})",
            color=color,
        )
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    for axis in axes:
        axis.legend(fontsize=8, loc="best")
        axis.grid(alpha=0.2)
    fig.suptitle(f"{dataset.replace('_', ' ').title()} probability curves")
    fig.savefig(output_dir / f"{dataset}_curves.png", dpi=180)
    plt.close(fig)


def generate_reports(
    results: pd.DataFrame,
    probabilities: dict[str, dict[str, np.ndarray]],
    test_labels: dict[str, np.ndarray],
    output_dir: str | Path,
) -> None:
    """Write comparison charts, best-model reports, and probability curves."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    successful = results.loc[(results["status"] == "ok") & results["pr_auc"].notna()]
    if successful.empty:
        (root / "evaluation_summary.txt").write_text(
            "No model completed successfully. See overall_benchmark_results.csv for errors.\n",
            encoding="utf-8",
        )
        return

    sns.set_theme(style="whitegrid", context="notebook")
    for dataset, dataset_rows in successful.groupby("dataset", sort=False):
        dataset_rows = dataset_rows.sort_values("pr_auc", ascending=False)
        scores_by_model = probabilities.get(str(dataset), {})
        dataset_rows = dataset_rows[dataset_rows["model"].isin(scores_by_model)]
        if dataset_rows.empty:
            continue
        y_true = test_labels[str(dataset)]
        _save_pr_auc_chart(str(dataset), dataset_rows, root)
        _save_probability_curves(str(dataset), dataset_rows, y_true, scores_by_model, root)
        best = dataset_rows.iloc[0]
        best_name = str(best["model"])
        best_scores = scores_by_model[best_name]
        _save_confusion_matrix(
            str(dataset), best_name, y_true, best_scores, float(best["pr_auc"]), root
        )
        report = classification_report(
            y_true,
            best_scores >= 0.5,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
            digits=4,
        )
        lines = [
            f"Dataset: {dataset}",
            f"Best model by PR-AUC: {best_name}",
            f"PR-AUC: {best['pr_auc']:.6f}",
            f"ROC-AUC: {best['roc_auc']:.6f}",
            f"Training seconds: {best['training_seconds']:.3f}",
            "Classification report at threshold 0.5:",
            report,
            "All successful model metrics:",
            dataset_rows[
                ["model", "pr_auc", "roc_auc", "precision", "recall", "f1_score", "training_seconds"]
            ].to_string(index=False),
        ]
        (root / f"{dataset}_evaluation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")