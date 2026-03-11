import argparse
import ast
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import wfdb
import time


# PTB-XL superclass order used everywhere in this project.
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


@dataclass(frozen=True)
class SplitArrays:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    meta_test: pd.DataFrame


def _ensure_dirs(project_root: str) -> None:
    os.makedirs(os.path.join(project_root, "src"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "outputs", "results"), exist_ok=True)


def _print_download_instructions(data_dir: str) -> None:
    print("PTB-XL data location check:")
    print(f"  Expected under: {data_dir}")
    print("  Required paths:")
    print("   - ptbxl_database.csv")
    print("   - scp_statements.csv")
    print("   - records100/ (contains waveform files)")
    print()
    print("If you do NOT have the dataset yet (no login required):")
    print("  1) Download PTB-XL from PhysioNet and extract it")
    print("  2) Put its contents into the path above")
    print()
    print("Optional (Linux/macOS) wget example:")
    print("  wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/")
    print()


def _load_metadata(data_dir: str) -> pd.DataFrame:
    db_path = os.path.join(data_dir, "ptbxl_database.csv")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Missing {db_path}")
    df = pd.read_csv(db_path)
    return df


def _load_scp_statements(data_dir: str) -> pd.DataFrame:
    stm_path = os.path.join(data_dir, "scp_statements.csv")
    if not os.path.exists(stm_path):
        raise FileNotFoundError(f"Missing {stm_path}")
    st = pd.read_csv(stm_path, index_col=0)
    # Index is scp_code; diagnostic_class contains the 5 superclasses we want.
    return st


def _parse_scp_codes(s: str) -> dict:
    # `scp_codes` is a dict string like "{'NORM': 100.0, 'LVOLT': 0.0, ...}"
    if pd.isna(s) or s is None or str(s).strip() == "":
        return {}
    try:
        d = ast.literal_eval(s)
        if isinstance(d, dict):
            return d
    except Exception:
        pass
    return {}


def _labels_for_record(scp_codes: dict, scp_statements: pd.DataFrame) -> list[str]:
    labels: set[str] = set()
    for scp_code in scp_codes.keys():
        if scp_code not in scp_statements.index:
            continue
        diag_class = scp_statements.loc[scp_code, "diagnostic_class"]
        if isinstance(diag_class, str) and diag_class in CLASS_TO_IDX:
            labels.add(diag_class)
    return sorted(labels, key=lambda c: CLASS_TO_IDX[c])


def _single_label_from_multilabel(
    labels: list[str], scp_codes: dict, scp_statements: pd.DataFrame
) -> str | None:
    if not labels:
        return None

    # "Highest-likelihood label": aggregate likelihood across scp_codes
    # that map to each superclass, then take the max.
    scores = {c: 0.0 for c in CLASSES}
    for scp_code, likelihood in scp_codes.items():
        if scp_code not in scp_statements.index:
            continue
        diag_class = scp_statements.loc[scp_code, "diagnostic_class"]
        if isinstance(diag_class, str) and diag_class in scores:
            try:
                scores[diag_class] += float(likelihood)
            except Exception:
                continue
    best = max(labels, key=lambda c: (scores.get(c, 0.0), -CLASS_TO_IDX[c]))
    return best


def _load_record_waveform(data_dir: str, filename_lr: str) -> np.ndarray:
    # wfdb expects the record path WITHOUT extension, relative to cwd or absolute.
    record_path = os.path.join(data_dir, filename_lr)
    signal, _fields = wfdb.rdsamp(record_path)
    # PTB-XL records100 are 100Hz, 10 seconds, 12 leads => (1000, 12)
    if signal.ndim != 2 or signal.shape[1] != 12:
        raise ValueError(f"Unexpected waveform shape {signal.shape} for {record_path}")
    if signal.shape[0] != 1000:
        # Be strict: this project assumes fixed-length 10s @ 100Hz.
        raise ValueError(f"Unexpected sample length {signal.shape[0]} for {record_path}")
    return signal.astype(np.float32, copy=False)


def _normalize_per_lead(x_t_by_lead: np.ndarray) -> np.ndarray:
    # x shape: (1000, 12). Normalize per lead across time within this record.
    mu = x_t_by_lead.mean(axis=0, keepdims=True)
    sigma = x_t_by_lead.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return (x_t_by_lead - mu) / sigma


def _split_by_fold(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "strat_fold" not in df.columns:
        raise KeyError("ptbxl_database.csv missing 'strat_fold' column")
    train = df[df["strat_fold"].between(1, 8)].copy()
    val = df[df["strat_fold"] == 9].copy()
    test = df[df["strat_fold"] == 10].copy()
    return train, val, test


def _print_class_distribution(name: str, y: np.ndarray) -> None:
    counts = np.bincount(y, minlength=len(CLASSES))
    total = int(counts.sum())
    print(f"{name} class distribution (n={total}):")
    for i, c in enumerate(CLASSES):
        print(f"  {c:4s}: {int(counts[i])}")
    print()


def run_phase1(
    *,
    project_root: str,
    data_dir: str,
    single_label: bool = True,
    limit: int | None = None,
) -> SplitArrays:
    _ensure_dirs(project_root)
    _print_download_instructions(data_dir)

    df = _load_metadata(data_dir)
    scp_statements = _load_scp_statements(data_dir)

    # Parse labels once; keep both multi-label list and single-label choice.
    multi_labels: list[list[str]] = []
    single_labels: list[str | None] = []

    for _, row in df.iterrows():
        scp_codes = _parse_scp_codes(row["scp_codes"])
        labels = _labels_for_record(scp_codes, scp_statements)
        multi_labels.append(labels)
        single_labels.append(_single_label_from_multilabel(labels, scp_codes, scp_statements))

    df = df.copy()
    df["superclass_labels"] = multi_labels
    df["superclass_label"] = single_labels

    if single_label:
        # Keep only rows with at least one superclass label.
        df = df[~df["superclass_label"].isna()].copy()
        df["y"] = df["superclass_label"].map(CLASS_TO_IDX).astype(int)
    else:
        raise NotImplementedError(
            "Multi-label experiments are supported in metadata (superclass_labels), "
            "but PHASE 1 artifacts are specified as single-label y arrays."
        )

    train_df, val_df, test_df = _split_by_fold(df)
    if limit is not None:
        train_df = train_df.iloc[:limit].copy()
        val_df = val_df.iloc[:limit].copy()
        test_df = test_df.iloc[:limit].copy()

    def build_split(split_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        xs: list[np.ndarray] = []
        ys: list[int] = []
        t0 = time.time()
        n = len(split_df)
        for _, row in split_df.iterrows():
            x = _load_record_waveform(data_dir, row["filename_lr"])
            x = _normalize_per_lead(x)  # (1000, 12)
            x = np.transpose(x, (1, 0))  # (12, 1000)
            xs.append(x)
            ys.append(int(row["y"]))
            if len(xs) % 500 == 0:
                dt = time.time() - t0
                rate = len(xs) / max(dt, 1e-6)
                print(f"  Loaded {len(xs)}/{n} records... ({rate:.1f} rec/s)", flush=True)
        X = np.stack(xs, axis=0).astype(np.float32, copy=False)  # (N, 12, 1000)
        y = np.asarray(ys, dtype=np.int64)
        return X, y

    print(f"Building TRAIN split ({len(train_df)} records)...", flush=True)
    X_train, y_train = build_split(train_df)
    print(f"Building VAL split ({len(val_df)} records)...", flush=True)
    X_val, y_val = build_split(val_df)
    print(f"Building TEST split ({len(test_df)} records)...", flush=True)
    X_test, y_test = build_split(test_df)

    out_dir = os.path.join(project_root, "outputs", "results")
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "X_val.npy"), X_val)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "y_val.npy"), y_val)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)

    meta_test = test_df[["ecg_id", "age", "sex", "strat_fold", "superclass_label"]].copy()
    meta_test = meta_test.rename(columns={"superclass_label": "true_label"})
    meta_test.to_csv(os.path.join(out_dir, "meta_test.csv"), index=False)

    print("Saved arrays:")
    print(f"  X_train.npy: {X_train.shape} (N, 12, 1000)")
    print(f"  X_val.npy:   {X_val.shape} (N, 12, 1000)")
    print(f"  X_test.npy:  {X_test.shape} (N, 12, 1000)")
    print(f"  y_train.npy: {y_train.shape} (N,)")
    print(f"  y_val.npy:   {y_val.shape} (N,)")
    print(f"  y_test.npy:  {y_test.shape} (N,)")
    print(f"  meta_test.csv: {meta_test.shape} rows x cols")
    print()

    _print_class_distribution("TRAIN", y_train)
    _print_class_distribution("VAL", y_val)
    _print_class_distribution("TEST", y_test)

    return SplitArrays(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        meta_test=meta_test,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PHASE 1: PTB-XL data pipeline (100Hz).")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path containing ptbxl_database.csv, scp_statements.csv, records100/",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root where outputs/ will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit per split for quick smoke tests.",
    )
    args = parser.parse_args()

    run_phase1(
        project_root=args.project_root,
        data_dir=args.data_dir,
        single_label=True,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

