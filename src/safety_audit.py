"""
Phase 3: Safety Audit — danger-weighted error analysis for ECG classifier.

Loads best checkpoint, runs test inference, computes:
- Standard metrics (accuracy, macro F1, macro AUC, per-class F1)
- Raw confusion matrix (saved as .npy)
- Danger-weighted error rate (DWE) and per-class DWE using the clinical danger matrix
- Critical error breakdown (danger=3: MI↔NORM)
- Subgroup analysis: DWE by age group and sex (saved as .csv)
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Allow running both `python src/safety_audit.py` and `python -m src.safety_audit`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import ResNet1D_12Lead


CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
NUM_CLASSES = len(CLASSES)

# ---------------------------------------------------------------------------
# Clinical Danger Matrix (hardcoded — core contribution of the project)
# danger_matrix[true_class_idx][pred_class_idx] = severity (0=correct, 1=minor, 2=moderate, 3=critical)
# Order: NORM=0, MI=1, STTC=2, CD=3, HYP=4
#
# Rationale (documented in comments):
# - MI predicted as NORM = 3: missed heart attack, directly fatal
# - NORM predicted as MI = 3: unnecessary cath lab activation, procedural risk + massive resource cost
# - STTC predicted as NORM = 2: missed ischemia, time-sensitive treatment
# - MI predicted as STTC = 2: partial miss, still concerning
# - CD/HYP errors = 1: serious but rarely immediately fatal
# ---------------------------------------------------------------------------
DANGER_MATRIX = np.array(
    [
        [0, 3, 2, 1, 1],   # True NORM  -> Pred NORM, MI, STTC, CD, HYP
        [3, 0, 2, 1, 1],   # True MI
        [2, 2, 0, 1, 1],   # True STTC
        [1, 1, 1, 0, 1],   # True CD
        [1, 1, 1, 1, 0],   # True HYP
    ],
    dtype=np.float64,
)


def load_model_and_predict(
    checkpoint_path: str,
    X_test: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load best checkpoint and run inference on test set. Returns y_true, y_pred, probs."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"]
    model = ResNet1D_12Lead(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    X = torch.from_numpy(X_test.astype(np.float32))
    batch_size = 64
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size].to(device)
            logits = model(batch)
            all_logits.append(logits.cpu())
    logits = torch.cat(all_logits, dim=0)
    probs = torch.softmax(logits, dim=1).numpy()
    y_pred = np.argmax(probs, axis=1)
    return y_pred, probs


def run_safety_audit(
    results_dir: str,
    models_dir: str,
) -> None:
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(models_dir, "ResNet1D_12Lead_best.pt")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

    # Load test data
    X_test = np.load(os.path.join(results_dir, "X_test.npy"))
    y_test = np.load(os.path.join(results_dir, "y_test.npy"))
    y_true = y_test
    N = len(y_true)

    # Step 1 — Standard evaluation
    print("Loading model and running test inference...", flush=True)
    y_pred, probs = load_model_and_predict(checkpoint_path, X_test, device)

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)))
    macro_auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))

    print(f"Test accuracy:  {accuracy:.4f}")
    print(f"Test macro F1:  {macro_f1:.4f}")
    print(f"Test macro AUC: {macro_auc:.4f}")
    print("Per-class F1:")
    for c, f1 in zip(CLASSES, per_class_f1):
        print(f"  {c:4s}: {float(f1):.4f}")

    # Step 2 — Confusion matrix (raw counts)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        confusion[int(t), int(p)] += 1
    np.save(os.path.join(results_dir, "confusion_matrix.npy"), confusion)
    print(f"\nConfusion matrix saved to {results_dir}/confusion_matrix.npy")

    # Step 3 — Danger-weighted error rate
    # DWE = sum_ij confusion[i][j] * danger_matrix[i][j] / N
    weighted_sum = np.sum(confusion * DANGER_MATRIX)
    dwe = float(weighted_sum / N)

    # Per-class DWE: for each true class i, average danger of (true=i) samples
    # = sum_j confusion[i][j] * danger_matrix[i][j] / count_true_i
    per_class_dwe = {}
    for i, c in enumerate(CLASSES):
        count_i = int(confusion[i, :].sum())
        if count_i == 0:
            per_class_dwe[c] = 0.0
        else:
            d_i = np.sum(confusion[i, :] * DANGER_MATRIX[i, :]) / count_i
            per_class_dwe[c] = float(d_i)

    # Critical errors (danger = 3): MI→NORM and NORM→MI
    # danger_matrix[1][0] = 3 (MI true, NORM pred), danger_matrix[0][1] = 3 (NORM true, MI pred)
    mi_as_norm = int(confusion[1, 0])
    norm_as_mi = int(confusion[0, 1])
    critical_count = mi_as_norm + norm_as_mi
    critical_rate = critical_count / N if N else 0.0

    safety_results = {
        "model": "ResNet1D_12Lead",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_auc": macro_auc,
        "dwe": dwe,
        "per_class_dwe": per_class_dwe,
        "critical_errors": {
            "count": critical_count,
            "rate": critical_rate,
            "breakdown": {
                "MI→NORM": mi_as_norm,
                "NORM→MI": norm_as_mi,
            },
        },
    }
    out_path = os.path.join(results_dir, "safety_audit_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(safety_results, f, indent=2)
    print(f"\nSafety audit results saved to {out_path}")
    print(f"DWE: {dwe:.4f}")
    print(f"Critical errors: {critical_count} ({critical_rate:.4f}) — MI→NORM: {mi_as_norm}, NORM→MI: {norm_as_mi}")

    # Step 4 — Subgroup analysis (age, sex) using meta_test.csv
    meta_path = os.path.join(results_dir, "meta_test.csv")
    if not os.path.isfile(meta_path):
        print(f"Warning: {meta_path} not found; skipping subgroup analysis.")
        return

    meta = pd.read_csv(meta_path)
    if len(meta) != N:
        raise ValueError(f"meta_test.csv rows ({len(meta)}) != test set size ({N})")

    meta = meta.copy()
    meta["y_true"] = y_true
    meta["y_pred"] = y_pred

    # Map true/pred to danger for each row
    meta["danger"] = [DANGER_MATRIX[int(t), int(p)] for t, p in zip(y_true, y_pred)]

    # Age groups: <40, 40-60, 60-75, >75 (NaN -> "unknown")
    def age_group(a):
        if pd.isna(a):
            return "unknown"
        a = float(a)
        if a < 40:
            return "<40"
        if a < 60:
            return "40-60"
        if a <= 75:
            return "60-75"
        return ">75"

    meta["age_group"] = meta["age"].apply(age_group)
    meta["sex_label"] = meta["sex"].map({0: "female", 1: "male"}).fillna("unknown")

    rows = []

    # By age group
    for ag in ["<40", "40-60", "60-75", ">75"]:
        sub = meta[meta["age_group"] == ag]
        n = len(sub)
        if n == 0:
            rows.append({"subgroup_type": "age", "subgroup": ag, "n": 0, "dwe": np.nan})
        else:
            dwe_sub = sub["danger"].mean()
            rows.append({"subgroup_type": "age", "subgroup": ag, "n": n, "dwe": float(dwe_sub)})

    # By sex
    for sx in ["female", "male"]:
        sub = meta[meta["sex_label"] == sx]
        n = len(sub)
        if n == 0:
            rows.append({"subgroup_type": "sex", "subgroup": sx, "n": 0, "dwe": np.nan})
        else:
            dwe_sub = sub["danger"].mean()
            rows.append({"subgroup_type": "sex", "subgroup": sx, "n": n, "dwe": float(dwe_sub)})

    subgroup_df = pd.DataFrame(rows)
    subgroup_path = os.path.join(results_dir, "subgroup_analysis.csv")
    subgroup_df.to_csv(subgroup_path, index=False)
    print(f"\nSubgroup analysis saved to {subgroup_path}")
    print(subgroup_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Safety audit — DWE and subgroup analysis.")
    parser.add_argument("--results-dir", type=str, default="outputs/results")
    parser.add_argument("--models-dir", type=str, default="outputs/models")
    args = parser.parse_args()
    run_safety_audit(results_dir=args.results_dir, models_dir=args.models_dir)


if __name__ == "__main__":
    main()
