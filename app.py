import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

# =========================
# Page config (LIGHT only)
# =========================
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

BG = "#F6F8FC"
CARD = "#FFFFFF"
TEXT = "#111827"
MUTED = "rgba(17,24,39,0.65)"
BORDER = "rgba(15,23,42,0.08)"

st.markdown(
    f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
    background: {BG};
}}
.block-container {{
    padding-top: 1.1rem;
    padding-bottom: 1.5rem;
}}
#MainMenu {{visibility:hidden;}}
footer {{visibility:hidden;}}

section[data-testid="stSidebar"] > div {{
    border-right: 1px solid {BORDER};
}}

.card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 16px 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}}
.small {{
    color: {MUTED};
    font-size: 12px;
}}
</style>
""",
    unsafe_allow_html=True,
)

def kpi_card(title, value, subtitle=""):
    st.markdown(
        f"""
<div class="card">
  <div style="color:{MUTED}; font-weight:600; font-size:14px;">{title}</div>
  <div style="color:{TEXT}; font-weight:800; font-size:28px; margin-top:6px;">{value}</div>
  <div style="color:{MUTED}; font-size:12px; margin-top:6px;">{subtitle}</div>
</div>
""",
        unsafe_allow_html=True,
    )

# =========================
# Single-folder project paths
# =========================
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data.csv"

@st.cache_data(show_spinner=False)
def load_csv_with_fallback(path: Path) -> pd.DataFrame:
    # Encoding fallback to fix UnicodeDecodeError
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    # last attempt (raise real error)
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    # Same fallback for uploaded files
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(uploaded_file, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(uploaded_file)

# =========================
# Model pipeline
# =========================
def build_pipeline(n_estimators: int, max_depth: int | None, random_state: int):
    num_sel = selector(dtype_include=np.number)
    cat_sel = selector(dtype_exclude=np.number)

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_sel),
            ("cat", cat_pipe, cat_sel),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        max_depth=max_depth,
    )

    return Pipeline(steps=[("prep", pre), ("model", clf)])

def detect_target(df: pd.DataFrame) -> str:
    if "Machine failure" in df.columns:
        return "Machine failure"
    for c in df.columns:
        if c.strip().lower() in {"machine failure", "machine_failure", "failure"}:
            return c
    return df.columns[-1]

def columns_to_drop_for_model(df: pd.DataFrame, target: str) -> list[str]:
    # Avoid leakage/IDs if present (AI4I commonly includes these)
    drop = []
    for c in df.columns:
        cl = c.strip().lower()
        if cl in {"udi", "product id", "product_id", "id"}:
            drop.append(c)
        if c in {"TWF", "HDF", "PWF", "OSF", "RNF"}:
            drop.append(c)
    if target in drop:
        drop.remove(target)
    return drop

@st.cache_resource(show_spinner=False)
def train_eval_cached(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
    n_estimators: int,
    max_depth: int | None,
):
    df = df.copy().dropna(axis=0, how="all")

    # Coerce target to 0/1 int if possible
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].astype(str).str.strip()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    drop_cols = columns_to_drop_for_model(df, target_col)
    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y if y.nunique() == 2 else None,
    )

    pipe = build_pipeline(n_estimators=int(n_estimators), max_depth=max_depth, random_state=int(random_state))
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps["model"], "predict_proba") else None

    roc_auc = roc_auc_score(y_test, proba) if proba is not None and y_test.nunique() == 2 else np.nan
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred)

    roc = (None, None, None)
    pr = (None, None, None)
    ap = np.nan
    if proba is not None and y_test.nunique() == 2:
        fpr, tpr, thr = roc_curve(y_test, proba)
        roc = (fpr, tpr, thr)

        prec, rec, thr2 = precision_recall_curve(y_test, proba)
        pr = (prec, rec, thr2)

        ap = average_precision_score(y_test, proba)

    fi = pd.DataFrame({"feature": [], "importance": []})
    try:
        feature_names = pipe.named_steps["prep"].get_feature_names_out()
        importances = pipe.named_steps["model"].feature_importances_
        fi = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception:
        pass

    return pipe, X, X_test, y_test, pred, proba, roc_auc, report, cm, fi, roc, pr, ap

# =========================
# Header
# =========================
st.markdown("## Predictive Maintenance Dashboard")
st.caption(
    "An interactive dashboard that explains machine failure risk, explores key patterns, and evaluates a Random Forest classifier."
)
st.write("")

# =========================
# Sidebar
# =========================
st.sidebar.title("Controls")
page = st.sidebar.radio("Navigate", ["Overview", "EDA", "Model", "Predict"], index=0)

st.sidebar.divider()
st.sidebar.subheader("Data")

use_repo_data = st.sidebar.checkbox("Load data.csv on startup", value=True)

df = None
source_label = ""

if use_repo_data:
    if not DATA_PATH.exists():
        st.error("data.csv not found in the same folder as app.py. Place it next to app.py and rerun.")
        st.stop()
    df = load_csv_with_fallback(DATA_PATH)
    source_label = "Loaded: data.csv"
else:
    upload = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if upload is None:
        st.info("Upload a CSV or enable data.csv loading.")
        st.stop()
    df = load_uploaded_csv(upload)
    source_label = "Loaded: upload"

target_default = detect_target(df)
target_col = st.sidebar.selectbox(
    "Target column",
    options=df.columns.tolist(),
    index=df.columns.tolist().index(target_default),
)

st.sidebar.divider()
st.sidebar.subheader("Model")

test_size = st.sidebar.slider("Test split", 0.1, 0.4, 0.2, step=0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=10_000, value=42)
n_estimators = st.sidebar.slider("Trees (n_estimators)", 50, 600, 200, step=50)
depth_choice = st.sidebar.selectbox("Max depth", ["None", "10", "20", "30"], index=0)
max_depth = None if depth_choice == "None" else int(depth_choice)

threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, step=0.05)
run_train = st.sidebar.button("Train / Refresh", type="primary")

st.caption(f"Data source: {source_label}")

# =========================
# KPI row
# =========================
n_rows, n_cols = df.shape
fail_rate = np.nan
ytemp = pd.to_numeric(df[target_col], errors="coerce")
if ytemp.notna().any():
    fail_rate = float((ytemp.fillna(0) > 0).mean() * 100)

k1, k2, k3, k4 = st.columns(4, gap="large")
with k1:
    kpi_card("Rows", f"{n_rows:,}", "Records")
with k2:
    kpi_card("Columns", f"{n_cols:,}", "Features + target")
with k3:
    kpi_card("Failure rate", f"{fail_rate:.2f}%" if np.isfinite(fail_rate) else "—", "Share of failures")
with k4:
    kpi_card("Target", target_col, "Prediction target")

st.write("")

# =========================
# Pages
# =========================
if page == "Overview":
    left, right = st.columns([1.2, 1.0], gap="large")
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Project Summary")
        st.write(
            """
- Failure frequency and basic dataset profile  
- Feature patterns linked to failure risk  
- Model evaluation metrics and error analysis  
- Interactive prediction tool for scenario testing
            """.strip()
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Data Overview")
        st.dataframe(df.head(30), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "EDA":
    st.markdown("### Exploratory Data Analysis")

    a, b = st.columns([1.1, 1.0], gap="large")
    with a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Failure Distribution")
        y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
        dist = pd.DataFrame({"Failure status": y.map({0: "No failure", 1: "Failure"})})
        fig = px.histogram(dist, x="Failure status")
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Missing Values (Top Columns)")
        miss = df.isna().sum().sort_values(ascending=False)
        miss = miss[miss > 0].head(12)
        if len(miss) == 0:
            st.success("No missing values detected.")
        else:
            mdf = miss.reset_index()
            mdf.columns = ["column", "missing_count"]
            figm = px.bar(mdf, x="missing_count", y="column", orientation="h")
            figm.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(figm, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Correlation Heatmap (Numeric Features)")
    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.shape[1] < 2:
        st.info("Not enough numeric columns for correlation.")
    else:
        corr = num_df.corr(numeric_only=True)
        figc = px.imshow(corr, aspect="auto")
        figc.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(figc, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Feature Distribution Explorer")
    st.caption("Select a numeric feature to view its distribution.")
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target_col]
    if not num_cols:
        st.info("No numeric feature columns detected.")
    else:
        col = st.selectbox("Numeric feature", num_cols)
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        figd = px.histogram(s, nbins=40, labels={"value": col})
        figd.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(figd, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Model":
    st.markdown("### Model Performance")

    if not run_train:
        st.info("Click Train / Refresh in the sidebar to compute metrics.")
        st.stop()

    with st.spinner("Training model..."):
        pipe, X_all, X_test, y_test, pred, proba, roc_auc, report, cm, fi, roc, pr, ap = train_eval_cached(
            df=df,
            target_col=target_col,
            test_size=float(test_size),
            random_state=int(random_state),
            n_estimators=int(n_estimators),
            max_depth=max_depth,
        )

    rep_df = pd.DataFrame(report).T
    accuracy = float(report.get("accuracy", np.nan))

    m1, m2, m3, m4 = st.columns(4, gap="large")
    with m1:
        kpi_card("Accuracy", f"{accuracy:.3f}" if np.isfinite(accuracy) else "—", "Overall correctness")
    with m2:
        kpi_card("ROC-AUC", f"{roc_auc:.3f}" if np.isfinite(roc_auc) else "—", "Ranking quality")
    with m3:
        kpi_card("Average Precision", f"{ap:.3f}" if np.isfinite(ap) else "—", "Precision-Recall summary")
    with m4:
        kpi_card("Test rows", f"{len(y_test):,}", "Holdout size")

    st.write("")
    left, right = st.columns([1.2, 1.0], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Confusion Matrix")
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto")
        fig_cm.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Classification Report")
        show_cols = ["precision", "recall", "f1-score", "support"]
        rep_show = rep_df[[c for c in show_cols if c in rep_df.columns]].copy()
        st.dataframe(rep_show, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    a, b = st.columns([1.0, 1.0], gap="large")

    with a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("ROC Curve")
        if roc[0] is None:
            st.info("ROC curve not available (requires binary target and probabilities).")
        else:
            fpr, tpr, _ = roc
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name="ROC"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Baseline", mode="lines"))
            fig_roc.update_layout(
                height=360,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Feature Importance")
        if fi.empty:
            st.info("Feature importance could not be extracted.")
        else:
            topn = st.slider("Show top N", 5, 25, 12)
            fi_top = fi.head(topn).iloc[::-1]
            fig_fi = px.bar(fi_top, x="importance", y="feature", orientation="h")
            fig_fi.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_fi, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Predict":
    st.markdown("### Risk Prediction Tool")
    st.caption("Adjust inputs and view the predicted probability of failure.")

    if not run_train:
        st.info("Click Train / Refresh in the sidebar first.")
        st.stop()

    with st.spinner("Training model..."):
        pipe, X_all, X_test, y_test, pred, proba, roc_auc, report, cm, fi, roc, pr, ap = train_eval_cached(
            df=df,
            target_col=target_col,
            test_size=float(test_size),
            random_state=int(random_state),
            n_estimators=int(n_estimators),
            max_depth=max_depth,
        )

    drop_cols = columns_to_drop_for_model(df, target_col)
    X = df.drop(columns=[target_col] + drop_cols, errors="ignore").copy()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Input Selection")
    input_row = {}
    cols = st.columns(3, gap="large")

    for i, col in enumerate(X.columns):
        box = cols[i % 3]
        s = X[col]

        if pd.api.types.is_numeric_dtype(s):
            v = pd.to_numeric(s, errors="coerce").dropna()
            if len(v) == 0:
                input_row[col] = 0.0
                continue
            vmin = float(v.quantile(0.01))
            vmax = float(v.quantile(0.99))
            vmed = float(v.median())
            input_row[col] = box.slider(col, min_value=vmin, max_value=vmax, value=vmed)
        else:
            vals = s.dropna().astype(str).unique().tolist()
            vals = sorted(vals)[:200] if len(vals) > 200 else sorted(vals)
            input_row[col] = box.selectbox(col, vals) if vals else ""

    st.markdown("</div>", unsafe_allow_html=True)

    X_new = pd.DataFrame([input_row])

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction Result")

    if hasattr(pipe.named_steps["model"], "predict_proba"):
        p = float(pipe.predict_proba(X_new)[:, 1][0])
        decision = 1 if p >= threshold else 0
        st.success(f"Predicted failure probability: {p:.2%}")
        st.write(f"Decision threshold: {threshold:.2f}")
        st.write(f"Predicted class: {'Failure' if decision==1 else 'No failure'}")
        st.markdown("<div class='small'>Use this to test scenarios and see how risk changes.</div>", unsafe_allow_html=True)
    else:
        pred_cls = int(pipe.predict(X_new)[0])
        st.success(f"Predicted class: {pred_cls}")

    st.markdown("</div>", unsafe_allow_html=True)
