"""
ECG Clinical Safety Auditor — Streamlit app.
3 pages: ECG Viewer, Safety Audit Results, Subgroup Safety Explorer.
Dark theme. All heavy resources loaded once via @st.cache_resource.
"""

import os
import sys

# Project root for imports
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go

# PTB-XL data root: use "data" (Phase 1 layout) or "data/ptbxl" if you extracted PTB-XL into data/ptbxl/
DATA_DIR = os.environ.get("ECG_DATA_DIR", "data")

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

DANGER_MATRIX = [
    [0, 3, 2, 1, 1],
    [3, 0, 2, 1, 1],
    [2, 2, 0, 1, 1],
    [1, 1, 1, 0, 1],
    [1, 1, 1, 1, 0],
]
DANGER_LABELS = {0: "No error", 1: "Minor", 2: "Moderate", 3: "CRITICAL"}
DANGER_COLORS = {0: "green", 1: "yellow", 2: "orange", 3: "red"}

DANGER_SENTENCES = {
    3: "Critical — a heart attack was missed or a healthy patient flagged for emergency intervention.",
    2: "Moderate — an ischemic change was missed or misidentified. Requires urgent review.",
    1: "Minor — a low-risk classification error. Unlikely to cause immediate harm.",
}


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def load_test_arrays(results_dir: str):
    X_test = np.load(os.path.join(results_dir, "X_test.npy"))
    y_test = np.load(os.path.join(results_dir, "y_test.npy"))
    return X_test, y_test


@st.cache_resource
def load_meta_test(results_dir: str):
    path = os.path.join(results_dir, "meta_test.csv")
    return pd.read_csv(path)


@st.cache_resource
def load_confusion_matrix(results_dir: str):
    return np.load(os.path.join(results_dir, "confusion_matrix.npy"))


@st.cache_resource
def load_safety_audit_results(results_dir: str):
    path = os.path.join(results_dir, "safety_audit_results.json")
    with open(path, encoding="utf-8") as f:
        import json
        return json.load(f)


@st.cache_resource
def load_subgroup_analysis(results_dir: str):
    path = os.path.join(results_dir, "subgroup_analysis.csv")
    return pd.read_csv(path)


@st.cache_resource
def load_model_and_predict(models_dir: str, results_dir: str, device_type: str):
    from src.models import ResNet1D_12Lead
    device = torch.device(device_type)
    path = os.path.join(models_dir, "ResNet1D_12Lead_best.pt")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ResNet1D_12Lead(num_classes=5).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    X_test, _ = load_test_arrays(results_dir)
    X = torch.from_numpy(X_test.astype(np.float32))
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            batch = X[i : i + 64].to(device)
            all_logits.append(model(batch).cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    logits_t = torch.from_numpy(logits)
    probs = torch.softmax(logits_t, dim=1).numpy()
    y_pred = np.argmax(probs, axis=1)
    return y_pred, probs


@st.cache_resource
def load_figure_bytes(figures_dir: str, name: str):
    path = os.path.join(figures_dir, name)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return f.read()


@st.cache_resource
def load_ptbxl_database(data_dir: str):
    """Load PTB-XL database to get ecg_id -> filename_lr for waveform loading.
    Tries data_dir (e.g. 'data') then data_dir/ptbxl (e.g. 'data/ptbxl') so both layouts work."""
    for base in [data_dir, os.path.join(data_dir, "ptbxl")]:
        path = os.path.join(PROJECT_ROOT, base, "ptbxl_database.csv")
        if os.path.isfile(path):
            df = pd.read_csv(path)
            root = os.path.join(PROJECT_ROOT, base)
            return df[["ecg_id", "filename_lr"]], root
    return None, None


@st.cache_data
def load_waveform(data_root: str, filename_lr: str):
    """Load one record's waveform from PTB-XL records100."""
    import wfdb
    record_path = os.path.join(data_root, filename_lr)
    signal, _ = wfdb.rdsamp(record_path)
    return signal  # (1000, 12)


# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

def render_sidebar(safety: dict):
    st.sidebar.title("ECG Safety Auditor")
    st.sidebar.caption("Measuring not just accuracy — but the danger of mistakes.")
    st.sidebar.divider()
    st.sidebar.metric("Test Macro F1", f"{safety['macro_f1']:.3f}")
    st.sidebar.metric("Overall DWE", f"{safety['dwe']:.3f}")
    st.sidebar.metric("Critical Error Rate", f"{safety['critical_errors']['rate']:.2%}")
    st.sidebar.divider()
    st.sidebar.markdown("[PTB-XL Dataset](https://physionet.org/content/ptb-xl/1.0.3/)")
    st.sidebar.markdown("[GitHub](https://github.com/youssof20/ecg-safety-auditor)")
    st.sidebar.caption("_Research only. Not for clinical use._")


def page_ecg_viewer(meta: pd.DataFrame, y_pred: np.ndarray, probs: np.ndarray, y_test: np.ndarray,
                    db_and_root, results_dir: str):
    st.title("ECG Viewer")
    st.caption("Explore individual test-set ECGs and their model predictions.")

    # Indices where danger == 3 (critical errors)
    danger_3_mask = np.zeros(len(y_test), dtype=bool)
    for i in range(len(y_test)):
        t, p = int(y_test[i]), int(y_pred[i])
        if DANGER_MATRIX[t][p] == 3:
            danger_3_mask[i] = True
    critical_indices = np.where(danger_3_mask)[0].tolist()

    show_only_critical = st.toggle("Show only critical errors (danger = 3)", value=False)
    if show_only_critical and not critical_indices:
        st.warning("No critical errors in the test set.")
        return
    if show_only_critical:
        pool = critical_indices
    else:
        pool = list(range(len(meta)))

    # Session state for current index
    if "viewer_index" not in st.session_state:
        st.session_state.viewer_index = int(np.random.choice(pool))
    current = st.session_state.viewer_index
    if current not in pool:
        current = pool[0]
        st.session_state.viewer_index = current

    if st.button("Next patient"):
        st.session_state.viewer_index = int(np.random.choice(pool))
        st.rerun()

    row = meta.iloc[current]
    ecg_id = row["ecg_id"]
    true_idx = int(y_test[current])
    pred_idx = int(y_pred[current])
    correct = true_idx == pred_idx
    danger_level = DANGER_MATRIX[true_idx][pred_idx]

    col_left, col_right = st.columns([0.4, 0.6])

    with col_left:
        st.subheader("Patient")
        st.write(f"**ECG ID:** {ecg_id}")
        st.write(f"**Age:** {row['age']}")
        st.write(f"**Sex:** {'Male' if row.get('sex', 0) == 1 else 'Female'}")
        st.write("**True class:**")
        st.markdown(f"`{CLASSES[true_idx]}`")
        st.write("**Predicted class:**")
        if correct:
            st.success(f"✓ {CLASSES[pred_idx]}")
        else:
            st.error(f"✗ {CLASSES[pred_idx]}")
            color = DANGER_COLORS.get(danger_level, "gray")
            st.markdown(f"**Danger level:** :{color}[{DANGER_LABELS.get(danger_level, '')}]")
            st.caption(DANGER_SENTENCES.get(danger_level, ""))

        # Softmax bar chart (plotly)
        fig_probs = go.Figure(data=[go.Bar(x=CLASSES, y=probs[current], marker_color=["#e74c3c" if i == pred_idx else "#3498db" for i in range(5)])])
        fig_probs.update_layout(title="Softmax probabilities", xaxis_title="Class", yaxis_title="Prob", height=280, margin=dict(t=40, b=40), template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_probs, width="stretch")

    with col_right:
        db_df, data_root = db_and_root
        if db_df is None or data_root is None:
            st.error("PTB-XL database not found. Set ECG_DATA_DIR or place ptbxl_database.csv and records100/ under data/.")
            return
        match = db_df[db_df["ecg_id"] == ecg_id]
        if match.empty:
            st.error(f"No filename for ECG ID {ecg_id} in database.")
            return
        filename_lr = match["filename_lr"].iloc[0]
        try:
            signal = load_waveform(data_root, filename_lr)  # (1000, 12)
        except Exception as e:
            st.error(f"Could not load waveform: {e}")
            return
        # Stack 12 leads with offset
        n_samples = signal.shape[0]
        t = np.arange(n_samples) / 100.0  # 100 Hz -> seconds
        offset = 2.0
        fig = go.Figure()
        for lead in range(12):
            y = signal[:, lead] + (11 - lead) * offset
            fig.add_trace(go.Scatter(x=t, y=y, name=LEAD_NAMES[lead], line=dict(width=0.8)))
        fig.update_layout(
            title=f"ECG #{ecg_id} — True: {CLASSES[true_idx]} | Pred: {CLASSES[pred_idx]}",
            title_font_color="#e74c3c" if not correct else "#27ae60",
            xaxis_title="Time (s)",
            height=500,
            showlegend=False,
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            yaxis=dict(showticklabels=True, tickvals=[(11 - i) * offset for i in range(12)], ticktext=LEAD_NAMES, side="left"),
            margin=dict(l=60),
        )
        st.plotly_chart(fig, width="stretch")


def page_safety_audit_results(safety: dict, figures_dir: str):
    st.title("Safety Audit Results")
    st.caption("How dangerous are this model's mistakes?")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test Accuracy", f"{safety['accuracy']:.2%}")
    col2.metric("Macro F1", f"{safety['macro_f1']:.3f}")
    col3.metric("DWE", f"{safety['dwe']:.3f}")
    col4.metric("Critical Error Rate", f"{safety['critical_errors']['rate']:.2%}")

    n_test = 2158
    n_wrong = int(round(n_test * (1 - safety["accuracy"])))
    critical_count = safety["critical_errors"]["count"]
    critical_pct = 100 * critical_count / n_test
    st.info(
        f"Out of {n_test:,} test ECGs, {n_wrong:,} were misclassified. "
        f"{critical_count} of those errors were clinically critical (danger level 3) — "
        f"meaning a heart attack was missed or a healthy patient was sent for emergency cardiac intervention. "
        f"That is {critical_pct:.1f}% of all test ECGs."
    )

    st.subheader("Danger matrix")
    st.caption("Clinical danger of each misclassification type (row = true, col = pred).")
    # Build HTML table: red=3, orange=2, yellow=1, white=0
    colors = {0: "#ffffff", 1: "#ffff00", 2: "#ffa500", 3: "#ff0000"}
    rows = ["<tr><th></th>" + "".join(f"<th>{c}</th>" for c in CLASSES) + "</tr>"]
    for i, c in enumerate(CLASSES):
        cells = "".join(f'<td style="background-color:{colors[DANGER_MATRIX[i][j]]}; padding:6px; text-align:center;">{DANGER_MATRIX[i][j]}</td>' for j in range(5))
        rows.append(f"<tr><th>{c}</th>{cells}</tr>")
    st.markdown(f'<table style="border-collapse: collapse;">{"".join(rows)}</table>', unsafe_allow_html=True)

    st.subheader("Figures")
    fig_names = [
        ("danger_confusion_matrix.png", "Confusion matrix overlaid with clinical danger levels. Red borders mark critical errors."),
        ("dwe_by_class.png", "Per-class danger-weighted error rate. MI and NORM carry the highest danger."),
        ("critical_error_breakdown.png", "Of 88 critical errors: 61 were missed MIs, 27 were false MI alarms."),
        ("subgroup_dwe_age.png", "DWE increases monotonically with age. Patients over 75 face the most dangerous errors."),
        ("subgroup_dwe_sex.png", "Male patients receive significantly more dangerous misclassifications than female patients."),
    ]
    for name, cap in fig_names:
        data = load_figure_bytes(figures_dir, name)
        if data:
            st.image(data, caption=cap, width="stretch")
        else:
            st.caption(f"[Missing: {name}]")


def page_subgroup_explorer(meta: pd.DataFrame, y_pred: np.ndarray, y_test: np.ndarray,
                           safety: dict, subgroup_df: pd.DataFrame):
    st.title("Subgroup Safety Explorer")
    st.caption("Does the model make more dangerous errors for certain patients?")

    meta = meta.copy()
    meta["y_true"] = y_test
    meta["y_pred"] = y_pred
    meta["danger"] = [DANGER_MATRIX[int(t)][int(p)] for t, p in zip(y_test, y_pred)]

    def age_group(a):
        if pd.isna(a):
            return "unknown"
        a = float(a)
        if a < 40: return "<40"
        if a < 60: return "40-60"
        if a <= 75: return "60-75"
        return ">75"
    meta["age_group"] = meta["age"].apply(age_group)
    meta["sex_label"] = meta["sex"].map({0: "female", 1: "male"}).fillna("unknown")

    col1, col2 = st.columns(2)
    with col1:
        age_sel = st.selectbox("Age group", ["All", "<40", "40-60", "60-75", ">75"])
    with col2:
        sex_sel = st.selectbox("Sex", ["All", "Female", "Male"])

    sub = meta.copy()
    if age_sel != "All":
        sub = sub[sub["age_group"] == age_sel]
    if sex_sel != "All":
        sub = sub[sub["sex_label"] == sex_sel.lower()]

    n = len(sub)
    if n == 0:
        st.warning("No samples in this subgroup.")
        return
    dwe_sub = sub["danger"].mean()
    critical_count_sub = (sub["danger"] == 3).sum()
    overall_dwe = safety["dwe"]
    st.metric("Sample size", n)
    st.metric("DWE", f"{dwe_sub:.3f}")
    st.metric("Critical errors (danger=3)", int(critical_count_sub))
    if dwe_sub > overall_dwe:
        st.warning("⚠️ This subgroup has a higher-than-average danger-weighted error rate.")

    # Confusion matrix for subgroup
    conf_sub = np.zeros((5, 5), dtype=np.int64)
    for _, r in sub.iterrows():
        conf_sub[int(r["y_true"]), int(r["y_pred"])] += 1
    row_sums = conf_sub.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_norm = conf_sub.astype(float) / row_sums
    filter_desc = f"Age: {age_sel}, Sex: {sex_sel} (n={n})"
    fig_cm = go.Figure(data=go.Heatmap(z=cm_norm, x=CLASSES, y=CLASSES, colorscale="Blues", text=conf_sub, texttemplate="%{text}", textfont={"size": 10}))
    fig_cm.update_layout(title=f"Confusion Matrix — {filter_desc}", xaxis_title="Predicted", yaxis_title="True", height=400, template="plotly_dark")
    st.plotly_chart(fig_cm, width="stretch")

    # Per-class F1 for subgroup
    from sklearn.metrics import f1_score
    y_t = sub["y_true"].astype(int).values
    y_p = sub["y_pred"].astype(int).values
    f1_per = f1_score(y_t, y_p, average=None, labels=list(range(5)), zero_division=0)
    colors_f1 = ["#27ae60" if f > 0.6 else "#e67e22" if f > 0.4 else "#e74c3c" for f in f1_per]
    fig_f1 = go.Figure(data=[go.Bar(x=CLASSES, y=f1_per, marker_color=colors_f1)])
    fig_f1.update_layout(title="Per-class F1 (subgroup)", xaxis_title="Class", yaxis_title="F1", yaxis_range=[0, 1], height=350, template="plotly_dark")
    st.plotly_chart(fig_f1, width="stretch")

    st.divider()
    st.info(
        "**Key finding:** DWE increases monotonically with patient age (0.31 → 0.48 → 0.49 → 0.55). "
        "Male patients (DWE=0.54) receive more dangerous errors than female patients (DWE=0.41). "
        "This model should be validated separately on elderly and male subpopulations before clinical deployment."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(layout="wide", page_title="ECG Safety Auditor")

    results_dir = os.path.join(PROJECT_ROOT, "outputs", "results")
    figures_dir = os.path.join(PROJECT_ROOT, "outputs", "figures")
    models_dir = os.path.join(PROJECT_ROOT, "outputs", "models")

    if not os.path.isfile(os.path.join(results_dir, "safety_audit_results.json")):
        st.error("Safety audit results not found. Run Phase 3 first: python -m src.safety_audit")
        st.stop()
    safety = load_safety_audit_results(results_dir)

    # Dark theme
    st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    [data-testid="stSidebar"] { background-color: #262730; }
    </style>
    """, unsafe_allow_html=True)

    render_sidebar(safety)

    try:
        X_test, y_test = load_test_arrays(results_dir)
        meta = load_meta_test(results_dir)
        subgroup_df = load_subgroup_analysis(results_dir)
    except Exception as e:
        st.error(f"Failed to load results: {e}")
        st.stop()

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        y_pred, probs = load_model_and_predict(models_dir, results_dir, device_type)
    except Exception as e:
        st.error(f"Failed to load model or run inference: {e}")
        st.stop()

    db_and_root = load_ptbxl_database(DATA_DIR)

    page = st.sidebar.radio("Page", ["ECG Viewer", "Safety Audit Results", "Subgroup Safety Explorer"], label_visibility="collapsed")
    if page == "ECG Viewer":
        page_ecg_viewer(meta, y_pred, probs, y_test, db_and_root, results_dir)
    elif page == "Safety Audit Results":
        page_safety_audit_results(safety, figures_dir)
    else:
        page_subgroup_explorer(meta, y_pred, y_test, safety, subgroup_df)


if __name__ == "__main__":
    main()
