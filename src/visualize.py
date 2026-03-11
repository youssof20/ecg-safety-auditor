"""
Phase 4: Visualization — publication-style figures for the safety audit.

Reads outputs/results/ (confusion_matrix.npy, safety_audit_results.json, subgroup_analysis.csv)
and generates 5 figures in outputs/figures/.
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Danger matrix for overlay (same as safety_audit): danger_matrix[true][pred]
# 0=correct, 1=minor, 2=moderate, 3=critical
DANGER_MATRIX = np.array(
    [
        [0, 3, 2, 1, 1],
        [3, 0, 2, 1, 1],
        [2, 2, 0, 1, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 1, 0],
    ],
    dtype=np.int64,
)
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def fig1_danger_confusion_matrix(confusion: np.ndarray, out_path: str) -> None:
    """Confusion matrix heatmap normalized by true class, with danger overlay (red=3, orange=2)."""
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_norm = confusion.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150, facecolor="white")
    ax.set_facecolor("white")
    im = ax.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_yticks(np.arange(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            text = ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
            # Overlay border by danger level
            d = DANGER_MATRIX[i, j]
            if d == 3:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=3)
                ax.add_patch(rect)
            elif d == 2:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="orange", linewidth=2)
                ax.add_patch(rect)

    ax.set_title("Confusion Matrix with Clinical Danger Overlay")
    fig.tight_layout(pad=1.5)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig2_dwe_by_class(per_class_dwe: dict, out_path: str) -> None:
    """Horizontal bar chart: per-class DWE, ordered by DWE descending; color by level (red/orange/green)."""
    classes = [c for c in CLASSES if c in per_class_dwe]
    dwe_vals = [per_class_dwe[c] for c in classes]
    order = np.argsort(dwe_vals)[::-1]
    classes = [classes[i] for i in order]
    dwe_vals = [dwe_vals[i] for i in order]

    # Color: high (>=0.6) red, medium (0.35--0.6) orange, low (<0.35) green
    colors = []
    for v in dwe_vals:
        if v >= 0.6:
            colors.append("#c0392b")
        elif v >= 0.35:
            colors.append("#e67e22")
        else:
            colors.append("#27ae60")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150, facecolor="white")
    ax.set_facecolor("white")
    y_pos = np.arange(len(classes))
    bars = ax.barh(y_pos, dwe_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Danger-Weighted Error Rate")
    ax.set_title("Danger-Weighted Error Rate by True Class")
    ax.set_xlim(0, max(dwe_vals) * 1.15 if dwe_vals else 1)
    fig.tight_layout(pad=1.5)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig3_subgroup_dwe_age(subgroup_df: pd.DataFrame, out_path: str) -> None:
    """Grouped bar chart: DWE per age group; annotate sample size on each bar."""
    age_order = ["<40", "40-60", "60-75", ">75"]
    df = subgroup_df[subgroup_df["subgroup_type"] == "age"].set_index("subgroup")
    df = df.reindex(age_order).dropna(how="all")
    if df.empty:
        plt.close(plt.figure())
        return
    x = np.arange(len(df))
    dwe = df["dwe"].values
    n = df["n"].astype(int).values
    labels = df.index.tolist()

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150, facecolor="white")
    ax.set_facecolor("white")
    bars = ax.bar(x, dwe, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Danger-Weighted Error Rate")
    ax.set_xlabel("Age Group")
    ax.set_title("Model Safety by Patient Age Group")
    for i, (v, ni) in enumerate(zip(dwe, n)):
        ax.text(i, v, f" n={ni}", va="bottom", ha="center", fontsize=9)
    ax.set_ylim(0, max(dwe) * 1.2 if len(dwe) else 1)
    fig.tight_layout(pad=1.5)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig4_subgroup_dwe_sex(subgroup_df: pd.DataFrame, out_path: str) -> None:
    """Side-by-side bars: Male vs Female DWE."""
    df = subgroup_df[subgroup_df["subgroup_type"] == "sex"]
    if df.empty:
        plt.close(plt.figure())
        return
    df = df[df["subgroup"].isin(["female", "male"])]
    if df.empty:
        plt.close(plt.figure())
        return
    x = np.arange(len(df))
    dwe = df["dwe"].values
    labels = df["subgroup"].str.capitalize().tolist()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150, facecolor="white")
    ax.set_facecolor("white")
    colors = ["#9b59b6", "#3498db"]
    ax.bar(x, dwe, color=colors[: len(df)], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Danger-Weighted Error Rate")
    ax.set_title("Model Safety by Patient Sex")
    ax.set_ylim(0, max(dwe) * 1.2 if len(dwe) else 1)
    fig.tight_layout(pad=1.5)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig5_critical_error_breakdown(
    breakdown: dict,
    critical_count: int,
    critical_rate: float,
    n_test: int,
    accuracy: float,
    out_path: str,
) -> None:
    """Pie or bar of MI→NORM vs NORM→MI; annotate '% of test errors that are clinically critical'."""
    # Labels: use Unicode arrow from JSON or ASCII
    labels = list(breakdown.keys())
    sizes = list(breakdown.values())
    total_errors = int(round(n_test * (1 - accuracy)))
    pct_critical_of_errors = 100 * critical_count / total_errors if total_errors > 0 else 0

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150, facecolor="white")
    ax.set_facecolor("white")
    colors = ["#c0392b", "#e74c3c"]
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct="%1.0f", colors=colors[: len(sizes)], startangle=90)
    for t in texts:
        t.set_fontsize(10)
    ax.set_title("Critical Misclassification Breakdown")
    fig.text(0.5, 0.02, f"{pct_critical_of_errors:.1f}% of test errors are clinically critical (danger level 3).", ha="center", fontsize=10)
    fig.tight_layout(pad=1.5)
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def run_visualize(results_dir: str, figures_dir: str) -> None:
    os.makedirs(figures_dir, exist_ok=True)

    with open(os.path.join(results_dir, "safety_audit_results.json"), encoding="utf-8") as f:
        results = json.load(f)
    confusion = np.load(os.path.join(results_dir, "confusion_matrix.npy"))
    subgroup_df = pd.read_csv(os.path.join(results_dir, "subgroup_analysis.csv"))

    n_test = int(confusion.sum())
    accuracy = results["accuracy"]
    per_class_dwe = results["per_class_dwe"]
    critical = results["critical_errors"]
    breakdown = critical["breakdown"]
    critical_count = critical["count"]
    critical_rate = critical["rate"]

    fig1_danger_confusion_matrix(confusion, os.path.join(figures_dir, "danger_confusion_matrix.png"))
    fig2_dwe_by_class(per_class_dwe, os.path.join(figures_dir, "dwe_by_class.png"))
    fig3_subgroup_dwe_age(subgroup_df, os.path.join(figures_dir, "subgroup_dwe_age.png"))
    fig4_subgroup_dwe_sex(subgroup_df, os.path.join(figures_dir, "subgroup_dwe_sex.png"))
    fig5_critical_error_breakdown(
        breakdown,
        critical_count,
        critical_rate,
        n_test,
        accuracy,
        os.path.join(figures_dir, "critical_error_breakdown.png"),
    )
    print("Saved 5 figures to", figures_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Generate safety audit figures.")
    parser.add_argument("--results-dir", type=str, default="outputs/results")
    parser.add_argument("--figures-dir", type=str, default="outputs/figures")
    args = parser.parse_args()
    run_visualize(results_dir=args.results_dir, figures_dir=args.figures_dir)


if __name__ == "__main__":
    main()
