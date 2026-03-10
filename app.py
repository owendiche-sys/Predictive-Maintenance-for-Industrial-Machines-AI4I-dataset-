from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Predictive Maintenance Decision Support Dashboard",
    layout="wide",
)

# =========================
# Theme
# =========================
BG = "#F6F8FC"
CARD = "#FFFFFF"
TEXT = "#0F172A"
MUTED = "rgba(15,23,42,0.68)"
BORDER = "rgba(15,23,42,0.08)"
PRIMARY = "#2563EB"
PRIMARY_SOFT = "rgba(37,99,235,0.12)"
SUCCESS = "#059669"
WARNING = "#D97706"
DANGER = "#DC2626"

st.markdown(
    f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
    background: {BG};
}}
.block-container {{
    padding-top: 2.3rem;
    padding-bottom: 1.5rem;
    max-width: 1500px;
}}
#MainMenu {{
    visibility: hidden;
}}
footer {{
    visibility: hidden;
}}

section[data-testid="stSidebar"] > div {{
    border-right: 1px solid {BORDER};
}}

.hero {{
    background: linear-gradient(135deg, rgba(37,99,235,0.10), rgba(37,99,235,0.03));
    border: 1px solid rgba(37,99,235,0.14);
    border-radius: 22px;
    padding: 18px 20px;
    box-shadow: 0 14px 35px rgba(0,0,0,0.06);
}}

.card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 16px 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.06);
}}

.metric-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 15px 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
}}

.badge {{
    display: inline-block;
    background: {PRIMARY_SOFT};
    color: {PRIMARY};
    border: 1px solid rgba(37,99,235,0.15);
    border-radius: 999px;
    padding: 6px 12px;
    font-weight: 700;
    font-size: 12px;
    margin-bottom: 10px;
}}

.small {{
    color: {MUTED};
    font-size: 12px;
}}

.label-soft {{
    color: {MUTED};
    font-size: 13px;
    font-weight: 600;
}}

hr {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 10px 0 14px 0;
}}
</style>
""",
    unsafe_allow_html=True,
)


def kpi_card(title: str, value: str, subtitle: str = ""):
    st.markdown(
        f"""
<div class="metric-card">
  <div style="color:{MUTED}; font-weight:700; font-size:13px;">{title}</div>
  <div style="color:{TEXT}; font-weight:800; font-size:28px; margin-top:6px;">{value}</div>
  <div style="color:{MUTED}; font-size:12px; margin-top:6px;">{subtitle}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def section_card_start(title: str, subtitle: str = ""):
    sub = f'<div class="small" style="margin-top:4px;">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
<div class="card">
  <div style="font-size:20px; font-weight:800; color:{TEXT};">{title}</div>
  {sub}
  <hr>
""",
        unsafe_allow_html=True,
    )


def section_card_end():
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Single-folder project paths
# =========================
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data.csv"


# =========================
# Data loading
# =========================
@st.cache_data(show_spinner=False)
def load_csv_with_fallback(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=enc)
        except UnicodeDecodeError:
            continue
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)


# =========================
# Helpers
# =========================
def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def detect_target(df: pd.DataFrame) -> str:
    if "Machine failure" in df.columns:
        return "Machine failure"
    for c in df.columns:
        if c.strip().lower() in {"machine failure", "machine_failure", "failure"}:
            return c
    return df.columns[-1]


def columns_to_drop_for_model(df: pd.DataFrame, target: str) -> list[str]:
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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"Air temperature [K]", "Process temperature [K]"}.issubset(df.columns):
        df["Temp_Diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]

    if {"Torque [Nm]", "Rotational speed [rpm]"}.issubset(df.columns):
        df["Power_Proxy"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"]

    if {"Torque [Nm]", "Tool wear [min]"}.issubset(df.columns):
        df["Torque_Wear_Interaction"] = df["Torque [Nm]"] * df["Tool wear [min]"]

    if {"Tool wear [min]", "Rotational speed [rpm]"}.issubset(df.columns):
        df["Wear_to_Speed"] = df["Tool wear [min]"] / (df["Rotational speed [rpm]"] + 1)

    return df


def build_pipeline(model_name: str, n_estimators: int, max_depth: Optional[int], random_state: int):
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
            ("onehot", make_onehot_encoder()),
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

    if model_name == "Logistic Regression":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=int(random_state),
        )
    elif model_name == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=int(n_estimators),
            random_state=int(random_state),
            n_jobs=-1,
            max_depth=max_depth,
            class_weight="balanced",
        )
    elif model_name == "Extra Trees":
        clf = ExtraTreesClassifier(
            n_estimators=int(n_estimators),
            random_state=int(random_state),
            n_jobs=-1,
            max_depth=max_depth,
            class_weight="balanced",
        )
    elif model_name == "Gradient Boosting":
        clf = GradientBoostingClassifier(random_state=int(random_state))
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(steps=[("prep", pre), ("model", clf)])


def clean_feature_label(x: str) -> str:
    x = str(x)
    x = x.replace("num__", "")
    x = x.replace("cat__", "")
    x = x.replace("_", " ")
    return x


def risk_band(prob: float, threshold: float) -> str:
    if prob < threshold * 0.7:
        return "Low risk"
    elif prob < threshold:
        return "Medium risk"
    return "High risk"


def recommended_action(prob: float, threshold: float) -> str:
    band = risk_band(prob, threshold)
    if band == "High risk":
        return "Prioritise inspection or maintenance review soon."
    if band == "Medium risk":
        return "Monitor closely and consider scheduling a near-term check."
    return "No immediate intervention suggested under the current threshold."


# =========================
# Cached model training
# =========================
@st.cache_resource(show_spinner=False)
def train_eval_cached(
    df: pd.DataFrame,
    target_col: str,
    model_name: str,
    test_size: float,
    random_state: int,
    n_estimators: int,
    max_depth: Optional[int],
):
    df = df.copy().dropna(axis=0, how="all")
    df = engineer_features(df)

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

    pipe = build_pipeline(
        model_name=model_name,
        n_estimators=int(n_estimators),
        max_depth=max_depth,
        random_state=int(random_state),
    )
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
        feature_names = [clean_feature_label(x) for x in pipe.named_steps["prep"].get_feature_names_out()]
        model_obj = pipe.named_steps["model"]

        if hasattr(model_obj, "feature_importances_"):
            importances = model_obj.feature_importances_
            fi = (
                pd.DataFrame({"feature": feature_names, "importance": importances})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
        elif hasattr(model_obj, "coef_"):
            coefs = np.abs(model_obj.coef_[0])
            fi = (
                pd.DataFrame({"feature": feature_names, "importance": coefs})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
    except Exception:
        pass

    abs_err = None
    if proba is not None:
        y_arr = y_test.to_numpy()
        abs_err = np.abs(y_arr - proba)

    return pipe, X, X_test, y_test, pred, proba, roc_auc, report, cm, fi, roc, pr, ap, abs_err


@st.cache_data(show_spinner=False)
def compare_models_cached(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
    n_estimators: int,
    max_depth: Optional[int],
):
    models = [
        "Logistic Regression",
        "Random Forest",
        "Extra Trees",
        "Gradient Boosting",
    ]

    rows = []

    for model_name in models:
        _, _, _, y_test, pred, proba, roc_auc, _, _, _, _, _, ap, _ = train_eval_cached(
            df=df,
            target_col=target_col,
            model_name=model_name,
            test_size=test_size,
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
        )

        rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1": f1_score(y_test, pred, zero_division=0),
                "ROC-AUC": roc_auc,
                "Average Precision": ap,
            }
        )

    return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)


def threshold_table(y_true: pd.Series, proba: np.ndarray) -> pd.DataFrame:
    rows = []
    for thr in np.arange(0.10, 0.91, 0.05):
        pred_thr = (proba >= thr).astype(int)
        rows.append(
            {
                "Threshold": round(float(thr), 2),
                "Precision": precision_score(y_true, pred_thr, zero_division=0),
                "Recall": recall_score(y_true, pred_thr, zero_division=0),
                "F1": f1_score(y_true, pred_thr, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


# =========================
# Insight helpers
# =========================
def compute_numeric_group_stats(df: pd.DataFrame, target_col: str, feature: str) -> dict:
    d = df[[feature, target_col]].copy()
    d[target_col] = pd.to_numeric(d[target_col], errors="coerce").fillna(0).astype(int)
    d[feature] = pd.to_numeric(d[feature], errors="coerce")
    d = d.dropna(subset=[feature])

    if d.empty:
        return {}

    try:
        d["bin"] = pd.qcut(d[feature], q=8, duplicates="drop")
    except ValueError:
        d["bin"] = pd.cut(d[feature], bins=8)

    g = (
        d.groupby("bin", observed=True)
        .agg(n=(target_col, "size"), failure_rate=(target_col, "mean"), feat_median=(feature, "median"))
        .reset_index(drop=True)
        .sort_values("feat_median")
    )

    if g.empty:
        return {}

    first = g.iloc[0]
    last = g.iloc[-1]
    delta = (last["failure_rate"] - first["failure_rate"]) * 100

    return {
        "feature": feature,
        "first_rate": float(first["failure_rate"] * 100),
        "last_rate": float(last["failure_rate"] * 100),
        "delta_pp": float(delta),
        "table": g,
    }


def data_driven_insights(df: pd.DataFrame, target_col: str, drop_cols: list[str]) -> tuple[list[str], pd.DataFrame]:
    d = engineer_features(df.copy())
    y = pd.to_numeric(d[target_col], errors="coerce").fillna(0).astype(int)
    base_rate = float(y.mean() * 100) if len(y) else np.nan
    n = len(y)

    insights = []
    if np.isfinite(base_rate):
        insights.append(f"Failure rate is {base_rate:.2f}% across {n:,} records.")
    else:
        insights.append("Failure rate could not be computed for the current target column.")

    num_cols = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c]) and c not in drop_cols and c != target_col]
    results = []
    for f in num_cols[:60]:
        stats = compute_numeric_group_stats(d, target_col, f)
        if stats:
            results.append(
                {
                    "feature": stats["feature"],
                    "low_bin_failure_rate_pct": stats["first_rate"],
                    "high_bin_failure_rate_pct": stats["last_rate"],
                    "change_pp": stats["delta_pp"],
                }
            )

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values("change_pp", ascending=False).reset_index(drop=True)
        top = res_df.iloc[0]
        insights.append(
            f"The largest failure-rate shift appears in {top['feature']}, where failure rate rises from about "
            f"{top['low_bin_failure_rate_pct']:.2f}% to {top['high_bin_failure_rate_pct']:.2f}%."
        )
        for _, r in res_df.head(3).iterrows():
            insights.append(
                f"{r['feature']} shows a {r['change_pp']:.2f} percentage-point increase in failure rate from lower to higher ranges."
            )

    return insights, res_df


def model_driven_insights(fi: pd.DataFrame, roc_auc: float, ap: float, report: dict, model_name: str) -> list[str]:
    insights = []
    acc = float(report.get("accuracy", np.nan))
    if np.isfinite(acc):
        insights.append(f"{model_name} achieved an accuracy of {acc:.3f}.")
    if np.isfinite(roc_auc):
        insights.append(f"ROC-AUC is {roc_auc:.3f}, which shows strong ranking quality.")
    if np.isfinite(ap):
        insights.append(f"Average Precision is {ap:.3f}, which is useful for evaluating performance under class imbalance.")

    try:
        f1_fail = float(report.get("1", {}).get("f1-score", np.nan))
        rec_fail = float(report.get("1", {}).get("recall", np.nan))
        prec_fail = float(report.get("1", {}).get("precision", np.nan))
        if np.isfinite(f1_fail):
            insights.append(
                f"Failure-class F1 is {f1_fail:.3f}, with precision {prec_fail:.3f} and recall {rec_fail:.3f}."
            )
    except Exception:
        pass

    if fi is not None and not fi.empty:
        top_feats = fi.head(5)["feature"].tolist()
        insights.append(f"Top model drivers include: {', '.join(top_feats)}.")

    return insights


# =========================
# Header
# =========================
st.markdown(
    """
<div class="hero">
  <div class="badge">FEATURED PROJECT</div>
  <div style="font-size:32px; font-weight:900; color:#0F172A; line-height:1.15;">
    Predictive Maintenance Decision Support Dashboard
  </div>
  <div style="margin-top:10px; color:rgba(15,23,42,0.72); font-size:15px; max-width:980px;">
    An interactive app for exploring machine failure risk, comparing classification models, testing maintenance scenarios,
    and translating model outputs into practical recommendations.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")


# =========================
# Sidebar
# =========================
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

st.sidebar.title("Controls")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "EDA", "Insights", "Compare Models", "Model", "Predict", "Recommendations"],
    index=0,
)

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

model_name = st.sidebar.selectbox(
    "Model",
    ["Gradient Boosting", "Random Forest", "Extra Trees", "Logistic Regression"],
    index=0,
)

test_size = st.sidebar.slider("Test split", 0.1, 0.4, 0.2, step=0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=10_000, value=42)
n_estimators = st.sidebar.slider("Trees (n_estimators)", 50, 600, 300, step=50)
depth_choice = st.sidebar.selectbox("Max depth", ["None", "10", "20", "30"], index=0)
max_depth = None if depth_choice == "None" else int(depth_choice)

threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.45, step=0.05)

run_train = st.sidebar.button("Train / Refresh", type="primary")
if run_train:
    st.session_state.model_trained = True

st.sidebar.caption(source_label)


# =========================
# Pages
# =========================
if page == "Overview":
    df_view = engineer_features(df.copy())
    n_rows, n_cols = df.shape

    ytemp = pd.to_numeric(df[target_col], errors="coerce")
    fail_rate = float((ytemp.fillna(0) > 0).mean() * 100) if ytemp.notna().any() else np.nan

    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        kpi_card("Rows", f"{n_rows:,}", "Records")
    with c2:
        kpi_card("Columns", f"{n_cols:,}", "Original features + target")
    with c3:
        kpi_card("Failure rate", f"{fail_rate:.2f}%" if np.isfinite(fail_rate) else "—", "Share of failures")
    with c4:
        kpi_card("Selected model", model_name, "Current evaluation model")

    st.write("")

    left, right = st.columns([1.1, 1.0], gap="large")
    with left:
        section_card_start("Project Summary", "What this app is designed to do")
        st.write(
            """
- Compare multiple classification models for machine failure prediction
- Surface operating conditions associated with higher failure risk
- Evaluate ranking quality, recall, F1, and precision-recall trade-offs
- Test machine scenarios using an interactive failure-risk tool
- Support proactive maintenance decisions with interpretable outputs
            """.strip()
        )
        st.write("")
        st.markdown("**Engineered features used in modelling**")
        engineered = [c for c in ["Temp_Diff", "Power_Proxy", "Torque_Wear_Interaction", "Wear_to_Speed"] if c in df_view.columns]
        if engineered:
            st.write(", ".join(engineered))
        else:
            st.write("No engineered features were added from the current dataset schema.")
        section_card_end()

    with right:
        section_card_start("Data Preview", "First rows from the loaded dataset")
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)
        section_card_end()

elif page == "EDA":
    st.markdown("### Exploratory Data Analysis")

    a, b = st.columns([1.0, 1.0], gap="large")
    with a:
        section_card_start("Failure Distribution", "Class balance in the selected target")
        y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
        dist = pd.DataFrame({"Failure status": y.map({0: "No failure", 1: "Failure"})})
        fig = px.histogram(dist, x="Failure status")
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        section_card_end()

    with b:
        section_card_start("Missing Values", "Top columns by missing count")
        miss = df.isna().sum().sort_values(ascending=False)
        miss = miss[miss > 0].head(12)
        if len(miss) == 0:
            st.success("No missing values detected.")
        else:
            mdf = miss.reset_index()
            mdf.columns = ["column", "missing_count"]
            figm = px.bar(mdf, x="missing_count", y="column", orientation="h")
            figm.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(figm, use_container_width=True)
        section_card_end()

    st.write("")
    section_card_start("Correlation Heatmap", "Numeric features only")
    num_df = engineer_features(df.copy()).select_dtypes(include=[np.number]).copy()
    if num_df.shape[1] < 2:
        st.info("Not enough numeric columns for correlation.")
    else:
        corr = num_df.corr(numeric_only=True)
        figc = px.imshow(corr, aspect="auto")
        figc.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(figc, use_container_width=True)
    section_card_end()

    st.write("")
    x1, x2 = st.columns([1.0, 1.0], gap="large")
    df_eda = engineer_features(df.copy())
    num_cols = [c for c in df_eda.columns if pd.api.types.is_numeric_dtype(df_eda[c]) and c != target_col]

    with x1:
        section_card_start("Feature Distribution Explorer", "Pick a numeric feature to inspect")
        if not num_cols:
            st.info("No numeric feature columns detected.")
        else:
            col = st.selectbox("Numeric feature", num_cols)
            s = pd.to_numeric(df_eda[col], errors="coerce").dropna()
            figd = px.histogram(s, nbins=40, labels={"value": col})
            figd.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(figd, use_container_width=True)
        section_card_end()

    with x2:
        section_card_start("Feature vs Failure", "How the selected feature differs by class")
        if not num_cols:
            st.info("No numeric feature columns detected.")
        else:
            compare_col = st.selectbox("Numeric feature ", num_cols, key="compare_feature")
            plot_df = df_eda[[compare_col, target_col]].copy()
            plot_df[target_col] = pd.to_numeric(plot_df[target_col], errors="coerce").fillna(0).astype(int)
            plot_df["Failure status"] = plot_df[target_col].map({0: "No failure", 1: "Failure"})
            fig_box = px.box(plot_df, x="Failure status", y=compare_col, color="Failure status")
            fig_box.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
        section_card_end()

elif page == "Insights":
    st.markdown("### Insights")
    st.caption(
        "This section starts with data-driven findings, then adds model-driven interpretation after training."
    )

    drop_cols = columns_to_drop_for_model(df, target_col)

    data_ins, shift_table = data_driven_insights(df, target_col, drop_cols)

    yv = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    base_rate = float(yv.mean() * 100) if len(yv) else np.nan
    failures = int(yv.sum()) if len(yv) else 0
    non_fail = int((yv == 0).sum()) if len(yv) else 0

    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        kpi_card("Baseline failure rate", f"{base_rate:.2f}%" if np.isfinite(base_rate) else "—", "Across all records")
    with c2:
        kpi_card("Failures", f"{failures:,}", "Failure cases")
    with c3:
        kpi_card("Non-failures", f"{non_fail:,}", "Normal cases")
    with c4:
        kpi_card("Records", f"{len(yv):,}", "Used for insights")

    st.write("")
    section_card_start("Insights (data-driven)", "Plain-language takeaways from the dataset itself")
    for item in data_ins[:10]:
        st.write(f"- {item}")
    section_card_end()

    st.write("")
    section_card_start("Failure-rate shift by feature range", "Features ranked by how much failure rate rises from lower to higher ranges")
    if shift_table.empty:
        st.write("Not enough numeric feature data to compute failure-rate shifts.")
    else:
        topn = st.slider("Show top N features", 5, min(25, len(shift_table)), 12)
        show = shift_table.head(topn).copy()
        fig_shift = px.bar(
            show.iloc[::-1],
            x="change_pp",
            y="feature",
            orientation="h",
            labels={"change_pp": "Change in failure rate (percentage points)"},
        )
        fig_shift.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_shift, use_container_width=True)
        st.dataframe(show, use_container_width=True, hide_index=True)
    section_card_end()

    st.write("")
    section_card_start("Feature impact explorer", "See how failure rate changes across the range of a selected numeric feature")
    df_ins = engineer_features(df.copy())
    num_cols = [
        c
        for c in df_ins.columns
        if pd.api.types.is_numeric_dtype(df_ins[c]) and c not in drop_cols and c != target_col
    ]
    if not num_cols:
        st.write("No numeric feature columns available for this view.")
    else:
        feat_sel = st.selectbox("Numeric feature", num_cols, index=0)
        stats = compute_numeric_group_stats(df_ins, target_col, feat_sel)
        if not stats:
            st.write("Not enough valid values to compute bins for this feature.")
        else:
            g = stats["table"].copy()
            fig_bins = px.line(
                g,
                x="feat_median",
                y="failure_rate",
                markers=True,
                labels={"feat_median": "Feature bin median", "failure_rate": "Failure rate"},
            )
            fig_bins.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_bins, use_container_width=True)

            g2 = g.copy()
            g2["failure_rate_pct"] = (g2["failure_rate"] * 100).round(2)
            g2 = g2[["n", "failure_rate_pct", "feat_median"]].rename(columns={"feat_median": "bin_median"})
            st.dataframe(g2, use_container_width=True, hide_index=True)
    section_card_end()

    st.write("")
    section_card_start("Insights (model-driven)", "Train the selected model to unlock model diagnostics")
    if not st.session_state.model_trained:
        st.info("Click Train / Refresh in the sidebar to generate model-driven insights.")
        section_card_end()
        st.stop()

    with st.spinner("Training model for insights..."):
        pipe, X_all, X_test, y_test, pred, proba, roc_auc, report, cm, fi, roc, pr, ap, abs_err = train_eval_cached(
            df=df,
            target_col=target_col,
            model_name=model_name,
            test_size=float(test_size),
            random_state=int(random_state),
            n_estimators=int(n_estimators),
            max_depth=max_depth,
        )

    insights_m = model_driven_insights(fi, roc_auc, ap, report, model_name)

    accuracy = float(report.get("accuracy", np.nan))
    f1_fail = float(report.get("1", {}).get("f1-score", np.nan)) if isinstance(report, dict) else np.nan

    m1, m2, m3, m4 = st.columns(4, gap="large")
    with m1:
        kpi_card("Accuracy", f"{accuracy:.3f}" if np.isfinite(accuracy) else "—", "Overall correctness")
    with m2:
        kpi_card("ROC-AUC", f"{roc_auc:.3f}" if np.isfinite(roc_auc) else "—", "Ranking quality")
    with m3:
        kpi_card("Average Precision", f"{ap:.3f}" if np.isfinite(ap) else "—", "Precision-recall summary")
    with m4:
        kpi_card("Failure-class F1", f"{f1_fail:.3f}" if np.isfinite(f1_fail) else "—", "Class 1 performance")

    st.write("")
    for item in insights_m[:10]:
        st.write(f"- {item}")

    if fi is not None and not fi.empty:
        st.write("")
        st.subheader("Top Model Drivers")
        topn = st.slider("Show top N model drivers", 5, 25, 12)
        fi_top = fi.head(topn).iloc[::-1]
        fig_fi = px.bar(fi_top, x="importance", y="feature", orientation="h")
        fig_fi.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_fi, use_container_width=True)

    if abs_err is not None:
        st.write("")
        st.subheader("Where the model is less reliable")
        ae = pd.Series(abs_err, name="probability_error")
        fig_err = px.histogram(ae, nbins=60, labels={"value": "Probability error"})
        fig_err.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_err, use_container_width=True)

    section_card_end()

elif page == "Compare Models":
    st.markdown("### Model Comparison")
    st.caption("This view compares several classifiers so the final model choice is more justified.")

    if not st.session_state.model_trained:
        st.info("Click Train / Refresh in the sidebar to generate the comparison.")
        st.stop()

    with st.spinner("Comparing models..."):
        compare_df = compare_models_cached(
            df=df,
            target_col=target_col,
            test_size=float(test_size),
            random_state=int(random_state),
            n_estimators=int(n_estimators),
            max_depth=max_depth,
        )

    top_model = compare_df.iloc[0]["Model"]
    top_f1 = compare_df.iloc[0]["F1"]
    top_auc = compare_df.iloc[0]["ROC-AUC"]

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        kpi_card("Best model by F1", top_model, "Top holdout result")
    with c2:
        kpi_card("Top F1", f"{top_f1:.3f}", "Failure-class balance")
    with c3:
        kpi_card("Top ROC-AUC", f"{top_auc:.3f}", "Ranking quality")

    st.write("")
    left, right = st.columns([1.05, 1.0], gap="large")

    with left:
        section_card_start("Comparison Table", "Main holdout metrics")
        st.dataframe(compare_df, use_container_width=True, hide_index=True)
        section_card_end()

    with right:
        section_card_start("Compare by F1", "Higher is better")
        fig_cmp = px.bar(compare_df.sort_values("F1"), x="F1", y="Model", orientation="h")
        fig_cmp.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_cmp, use_container_width=True)
        section_card_end()

    st.write("")
    section_card_start("Compare by ROC-AUC", "Useful for ranking higher-risk cases")
    fig_auc = px.bar(compare_df.sort_values("ROC-AUC"), x="ROC-AUC", y="Model", orientation="h")
    fig_auc.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_auc, use_container_width=True)
    section_card_end()

elif page == "Model":
    st.markdown("### Model Performance")

    if not st.session_state.model_trained:
        st.info("Click Train / Refresh in the sidebar to compute metrics.")
        st.stop()

    with st.spinner("Training selected model..."):
        pipe, X_all, X_test, y_test, pred, proba, roc_auc, report, cm, fi, roc, pr, ap, abs_err = train_eval_cached(
            df=df,
            target_col=target_col,
            model_name=model_name,
            test_size=float(test_size),
            random_state=int(random_state),
            n_estimators=int(n_estimators),
            max_depth=max_depth,
        )

    rep_df = pd.DataFrame(report).T
    accuracy = float(report.get("accuracy", np.nan))
    f1_fail = float(report.get("1", {}).get("f1-score", np.nan))
    rec_fail = float(report.get("1", {}).get("recall", np.nan))
    prec_fail = float(report.get("1", {}).get("precision", np.nan))

    m1, m2, m3, m4 = st.columns(4, gap="large")
    with m1:
        kpi_card("Accuracy", f"{accuracy:.3f}" if np.isfinite(accuracy) else "—", "Overall correctness")
    with m2:
        kpi_card("Failure Precision", f"{prec_fail:.3f}" if np.isfinite(prec_fail) else "—", "Class 1 precision")
    with m3:
        kpi_card("Failure Recall", f"{rec_fail:.3f}" if np.isfinite(rec_fail) else "—", "Class 1 recall")
    with m4:
        kpi_card("Failure F1", f"{f1_fail:.3f}" if np.isfinite(f1_fail) else "—", "Class 1 F1")

    st.write("")

    left, right = st.columns([1.0, 1.0], gap="large")
    with left:
        section_card_start("Confusion Matrix", "How predictions split across classes")
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: No failure", "Actual: Failure"],
            columns=["Pred: No failure", "Pred: Failure"],
        )
        fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto")
        fig_cm.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_cm, use_container_width=True)
        section_card_end()

    with right:
        section_card_start("Classification Report", "Precision, recall, F1, and support")
        show_cols = ["precision", "recall", "f1-score", "support"]
        rep_show = rep_df[[c for c in show_cols if c in rep_df.columns]].copy()
        st.dataframe(rep_show, use_container_width=True)
        section_card_end()

    st.write("")
    a, b = st.columns([1.0, 1.0], gap="large")

    with a:
        section_card_start("ROC Curve", "Ranking performance across thresholds")
        if roc[0] is None:
            st.info("ROC curve not available.")
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
        section_card_end()

    with b:
        section_card_start("Precision-Recall Curve", "Useful when failures are relatively rare")
        if pr[0] is None:
            st.info("Precision-recall curve not available.")
        else:
            precision_vals, recall_vals, _ = pr
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=recall_vals, y=precision_vals, name="PR Curve"))
            fig_pr.update_layout(
                height=360,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Recall",
                yaxis_title="Precision",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_pr, use_container_width=True)
        section_card_end()

    st.write("")
    c, d = st.columns([1.0, 1.0], gap="large")

    with c:
        section_card_start("Feature Importance", "Main features used by the selected model")
        if fi.empty:
            st.info("Feature importance is not available for the current model.")
        else:
            topn = st.slider("Show top N features", 5, 25, 12)
            fi_top = fi.head(topn).iloc[::-1]
            fig_fi = px.bar(fi_top, x="importance", y="feature", orientation="h")
            fig_fi.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_fi, use_container_width=True)
        section_card_end()

    with d:
        section_card_start("Threshold Tuning", "How threshold choice changes precision, recall, and F1")
        if proba is None:
            st.info("Threshold tuning is not available without predicted probabilities.")
        else:
            thr_df = threshold_table(y_test, proba)
            fig_thr = go.Figure()
            fig_thr.add_trace(go.Scatter(x=thr_df["Threshold"], y=thr_df["Precision"], mode="lines+markers", name="Precision"))
            fig_thr.add_trace(go.Scatter(x=thr_df["Threshold"], y=thr_df["Recall"], mode="lines+markers", name="Recall"))
            fig_thr.add_trace(go.Scatter(x=thr_df["Threshold"], y=thr_df["F1"], mode="lines+markers", name="F1"))
            fig_thr.add_vline(x=threshold, line_dash="dash", line_color=PRIMARY)
            fig_thr.update_layout(
                height=360,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Decision threshold",
                yaxis_title="Score",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_thr, use_container_width=True)

            best_f1_row = thr_df.loc[thr_df["F1"].idxmax()]
            st.markdown(
                f"""
<div class="small">
Best threshold by F1: <b>{best_f1_row['Threshold']:.2f}</b> |
Precision: <b>{best_f1_row['Precision']:.3f}</b> |
Recall: <b>{best_f1_row['Recall']:.3f}</b> |
F1: <b>{best_f1_row['F1']:.3f}</b>
</div>
""",
                unsafe_allow_html=True,
            )
        section_card_end()

elif page == "Predict":
    st.markdown("### Risk Prediction Tool")
    st.caption("Adjust machine conditions and estimate failure risk using the selected model.")

    if not st.session_state.model_trained:
        st.info("Click Train / Refresh in the sidebar first.")
        st.stop()

    with st.spinner("Training selected model..."):
        pipe, X_all, X_test, y_test, pred, proba, roc_auc, report, cm, fi, roc, pr, ap, abs_err = train_eval_cached(
            df=df,
            target_col=target_col,
            model_name=model_name,
            test_size=float(test_size),
            random_state=int(random_state),
            n_estimators=int(n_estimators),
            max_depth=max_depth,
        )

    df_pred = engineer_features(df.copy())
    drop_cols = columns_to_drop_for_model(df_pred, target_col)
    X = df_pred.drop(columns=[target_col] + drop_cols, errors="ignore").copy()

    section_card_start("Input Selection", "Use realistic input values to simulate a machine state")
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
            if np.isclose(vmin, vmax):
                input_row[col] = box.number_input(col, value=float(vmed))
            else:
                input_row[col] = box.slider(col, min_value=vmin, max_value=vmax, value=vmed)
        else:
            vals = s.dropna().astype(str).unique().tolist()
            vals = sorted(vals)[:200] if len(vals) > 200 else sorted(vals)
            input_row[col] = box.selectbox(col, vals) if vals else ""

    section_card_end()

    X_new = pd.DataFrame([input_row])

    st.write("")
    left, right = st.columns([0.85, 1.15], gap="large")

    with left:
        section_card_start("Prediction Result", "Probability, risk band, and suggested action")
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            p = float(pipe.predict_proba(X_new)[:, 1][0])
            decision = 1 if p >= threshold else 0
            band = risk_band(p, threshold)
            action = recommended_action(p, threshold)

            color_map = {
                "Low risk": SUCCESS,
                "Medium risk": WARNING,
                "High risk": DANGER,
            }

            st.success(f"Predicted failure probability: {p:.2%}")
            st.write(f"Selected model: **{model_name}**")
            st.write(f"Decision threshold: **{threshold:.2f}**")
            st.markdown(
                f"<div style='font-weight:800; color:{color_map[band]}; font-size:18px;'>Risk band: {band}</div>",
                unsafe_allow_html=True,
            )
            st.write(f"Predicted class: **{'Failure' if decision == 1 else 'No failure'}**")
            st.write(f"Recommended action: **{action}**")

            st.markdown(
                "<div class='small'>This tool is best used to compare scenarios and prioritise higher-risk machine states.</div>",
                unsafe_allow_html=True,
            )
        else:
            pred_cls = int(pipe.predict(X_new)[0])
            st.success(f"Predicted class: {pred_cls}")
        section_card_end()

    with right:
        section_card_start("Risk Gauge", "Visual view of the predicted failure probability")
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            p = float(pipe.predict_proba(X_new)[:, 1][0])

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=p * 100,
                    number={"suffix": "%"},
                    title={"text": "Failure Probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": PRIMARY},
                        "steps": [
                            {"range": [0, threshold * 70], "color": "rgba(5,150,105,0.18)"},
                            {"range": [threshold * 70, threshold * 100], "color": "rgba(217,119,6,0.18)"},
                            {"range": [threshold * 100, 100], "color": "rgba(220,38,38,0.18)"},
                        ],
                        "threshold": {
                            "line": {"color": DANGER, "width": 4},
                            "thickness": 0.75,
                            "value": threshold * 100,
                        },
                    },
                )
            )
            gauge.update_layout(height=330, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(gauge, use_container_width=True)
        else:
            st.info("Gauge not available without predicted probabilities.")
        section_card_end()

elif page == "Recommendations":
    st.markdown("### Recommendations")
    st.caption("A practical summary of how the model can support predictive maintenance decisions.")

    c1, c2 = st.columns([1.0, 1.0], gap="large")

    with c1:
        section_card_start("Operational Recommendations", "How this tool could be used in practice")
        st.write(
            """
1. Use the selected model as a failure-risk ranking tool rather than only a yes/no classifier.
2. Prioritise machines with higher predicted probabilities for earlier inspection.
3. Use a lower threshold when the cost of missing a failure is high.
4. Monitor the strongest model drivers regularly and use them to support earlier intervention.
5. Retrain the model periodically with fresh machine data to keep it relevant.
            """.strip()
        )
        section_card_end()

    with c2:
        section_card_start("Project Positioning", "Why this is useful beyond a model score")
        st.write(
            """
This project is designed as a predictive maintenance decision-support workflow.

Instead of stopping at one classification score, it compares models, explores threshold trade-offs,
shows the most important drivers of failure risk, and provides an interactive scenario-testing tool.
That makes it more useful for maintenance planning, risk prioritisation, and operational communication.
            """.strip()
        )
        section_card_end()

    st.write("")
    section_card_start("Next Improvements", "Ways to make the project even stronger later")
    st.write(
        """
- Add SHAP explanations for more detailed local and global interpretability
- Introduce time-based features if sequential machine data becomes available
- Add cost-sensitive evaluation based on downtime and inspection costs
- Save the best trained model and expose a more production-style scoring workflow
- Compare model performance across different machine types or operating profiles
        """.strip()
    )
    section_card_end()