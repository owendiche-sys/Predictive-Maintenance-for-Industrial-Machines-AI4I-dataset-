from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
        padding-top: 2rem;
        padding-bottom: 1.5rem;
        max-width: 1460px;
    }}
    #MainMenu, footer {{
        visibility: hidden;
    }}
    section[data-testid="stSidebar"] > div {{
        border-right: 1px solid {BORDER};
    }}
    .hero {{
        background: linear-gradient(135deg, rgba(37,99,235,0.12), rgba(37,99,235,0.04));
        border: 1px solid rgba(37,99,235,0.14);
        border-radius: 24px;
        padding: 22px 22px 20px 22px;
        box-shadow: 0 14px 35px rgba(15,23,42,0.05);
    }}
    .badge {{
        display: inline-block;
        background: {PRIMARY_SOFT};
        color: {PRIMARY};
        border: 1px solid rgba(37,99,235,0.14);
        border-radius: 999px;
        padding: 6px 12px;
        font-weight: 800;
        font-size: 12px;
        letter-spacing: 0.02em;
        margin-bottom: 10px;
    }}
    .card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 20px;
        padding: 16px 16px;
        box-shadow: 0 10px 30px rgba(15,23,42,0.05);
    }}
    .metric-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 14px 16px;
        box-shadow: 0 10px 30px rgba(15,23,42,0.05);
        min-height: 122px;
    }}
    .metric-title {{
        color: {MUTED};
        font-size: 13px;
        font-weight: 700;
    }}
    .metric-value {{
        color: {TEXT};
        font-size: 29px;
        font-weight: 800;
        margin-top: 6px;
        line-height: 1.1;
    }}
    .metric-sub {{
        color: {MUTED};
        font-size: 12px;
        margin-top: 8px;
    }}
    .section-title {{
        color: {TEXT};
        font-size: 21px;
        font-weight: 850;
    }}
    .section-sub {{
        color: {MUTED};
        font-size: 13px;
        margin-top: 4px;
    }}
    .callout {{
        background: rgba(15,23,42,0.025);
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 12px 14px;
    }}
    .insight-item {{
        padding: 9px 0;
        border-bottom: 1px solid rgba(15,23,42,0.06);
    }}
    .insight-item:last-child {{
        border-bottom: none;
    }}
    .small {{
        color: {MUTED};
        font-size: 12px;
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

# =========================
# UI helpers
# =========================
def kpi_card(title: str, value: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_card_start(title: str, subtitle: str = "") -> None:
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="card">
            <div class="section-title">{title}</div>
            {sub}
            <hr>
        """,
        unsafe_allow_html=True,
    )


def section_card_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def display_insights(items: list[str], limit: Optional[int] = None) -> None:
    shown = items if limit is None else items[:limit]
    if not shown:
        st.write("No insights available for the current selection.")
        return
    for item in shown:
        st.markdown(f'<div class="insight-item">{item}</div>', unsafe_allow_html=True)


def clean_feature_label(value: str) -> str:
    value = str(value).replace("num__", "").replace("cat__", "")
    value = value.replace("_", " ")
    return value


# =========================
# Data paths
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
# Feature and target helpers
# =========================
def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def detect_target(df: pd.DataFrame) -> str:
    if "Machine failure" in df.columns:
        return "Machine failure"
    for col in df.columns:
        if col.strip().lower() in {"machine failure", "machine_failure", "failure"}:
            return col
    return df.columns[-1]


def columns_to_drop_for_model(df: pd.DataFrame, target: str) -> list[str]:
    drop_cols: list[str] = []
    for col in df.columns:
        lower = col.strip().lower()
        if lower in {"udi", "product id", "product_id", "id"}:
            drop_cols.append(col)
        if col in {"TWF", "HDF", "PWF", "OSF", "RNF"}:
            drop_cols.append(col)
    return [col for col in drop_cols if col != target]


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


def prepare_target_series(df: pd.DataFrame, target_col: str) -> pd.Series:
    y = pd.to_numeric(df[target_col], errors="coerce")
    return y.fillna(0).astype(int)


def base_failure_metrics(df: pd.DataFrame, target_col: str) -> dict:
    y = prepare_target_series(df, target_col)
    failures = int(y.sum())
    total = int(len(y))
    non_failures = int((y == 0).sum())
    rate = float(y.mean() * 100) if total else np.nan
    return {
        "records": total,
        "failures": failures,
        "non_failures": non_failures,
        "failure_rate": rate,
    }


def categorical_failure_rates(df: pd.DataFrame, target_col: str, max_rows: int = 12) -> pd.DataFrame:
    d = df.copy()
    d[target_col] = prepare_target_series(d, target_col)

    cat_cols = [col for col in d.columns if not pd.api.types.is_numeric_dtype(d[col]) and col != target_col]
    rows: list[dict] = []

    for col in cat_cols[:8]:
        tmp = d[[col, target_col]].copy()
        tmp[col] = tmp[col].astype(str).fillna("Missing")
        grp = tmp.groupby(col, dropna=False).agg(
            records=(target_col, "size"),
            failure_rate=(target_col, "mean"),
        )
        grp = grp[grp["records"] >= max(10, int(len(tmp) * 0.01))]
        if grp.empty:
            continue
        grp = grp.sort_values(["failure_rate", "records"], ascending=[False, False]).reset_index()
        best = grp.iloc[0]
        rows.append(
            {
                "feature": col,
                "category": str(best[col]),
                "records": int(best["records"]),
                "failure_rate_pct": float(best["failure_rate"] * 100),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["feature", "category", "records", "failure_rate_pct"])

    return pd.DataFrame(rows).sort_values("failure_rate_pct", ascending=False).head(max_rows).reset_index(drop=True)


def compute_numeric_group_stats(df: pd.DataFrame, target_col: str, feature: str) -> dict:
    d = df[[feature, target_col]].copy()
    d[target_col] = prepare_target_series(d, target_col)
    d[feature] = pd.to_numeric(d[feature], errors="coerce")
    d = d.dropna(subset=[feature])

    if d.empty or d[feature].nunique() < 4:
        return {}

    try:
        d["bin"] = pd.qcut(d[feature], q=8, duplicates="drop")
    except ValueError:
        d["bin"] = pd.cut(d[feature], bins=8)

    grouped = (
        d.groupby("bin", observed=True)
        .agg(
            records=(target_col, "size"),
            failure_rate=(target_col, "mean"),
            feature_median=(feature, "median"),
        )
        .reset_index(drop=True)
        .sort_values("feature_median")
    )

    if grouped.empty:
        return {}

    first = grouped.iloc[0]
    last = grouped.iloc[-1]

    return {
        "feature": feature,
        "low_failure_rate_pct": float(first["failure_rate"] * 100),
        "high_failure_rate_pct": float(last["failure_rate"] * 100),
        "change_pp": float((last["failure_rate"] - first["failure_rate"]) * 100),
        "table": grouped,
    }


def build_failure_shift_table(df: pd.DataFrame, target_col: str, drop_cols: list[str]) -> pd.DataFrame:
    engineered = engineer_features(df.copy())
    numeric_cols = [
        col
        for col in engineered.columns
        if pd.api.types.is_numeric_dtype(engineered[col]) and col not in drop_cols and col != target_col
    ]

    rows: list[dict] = []
    for feature in numeric_cols[:80]:
        stats = compute_numeric_group_stats(engineered, target_col, feature)
        if stats:
            rows.append(
                {
                    "feature": stats["feature"],
                    "low_range_failure_rate_pct": stats["low_failure_rate_pct"],
                    "high_range_failure_rate_pct": stats["high_failure_rate_pct"],
                    "change_pp": stats["change_pp"],
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["feature", "low_range_failure_rate_pct", "high_range_failure_rate_pct", "change_pp"]
        )

    return pd.DataFrame(rows).sort_values("change_pp", ascending=False).reset_index(drop=True)


def build_median_comparison_table(df: pd.DataFrame, target_col: str, drop_cols: list[str]) -> pd.DataFrame:
    d = engineer_features(df.copy())
    d[target_col] = prepare_target_series(d, target_col)
    num_cols = [
        col
        for col in d.columns
        if pd.api.types.is_numeric_dtype(d[col]) and col not in drop_cols and col != target_col
    ]

    rows: list[dict] = []
    for col in num_cols[:60]:
        tmp = d[[col, target_col]].copy()
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp.dropna()
        if tmp.empty:
            continue

        no_failure = tmp.loc[tmp[target_col] == 0, col]
        failure = tmp.loc[tmp[target_col] == 1, col]
        if no_failure.empty or failure.empty:
            continue

        no_med = float(no_failure.median())
        fail_med = float(failure.median())
        delta = fail_med - no_med
        rel = (delta / abs(no_med) * 100) if no_med != 0 else np.nan

        rows.append(
            {
                "feature": col,
                "non_failure_median": no_med,
                "failure_median": fail_med,
                "median_delta": delta,
                "relative_change_pct": rel,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["feature", "non_failure_median", "failure_median", "median_delta", "relative_change_pct"]
        )

    comparison = pd.DataFrame(rows)
    comparison["abs_relative_change_pct"] = comparison["relative_change_pct"].abs()
    comparison = comparison.sort_values("abs_relative_change_pct", ascending=False).drop(columns="abs_relative_change_pct")
    return comparison.reset_index(drop=True)


def generate_data_insights(
    df: pd.DataFrame,
    target_col: str,
    shift_table: pd.DataFrame,
    cat_table: pd.DataFrame,
    median_table: pd.DataFrame,
) -> list[str]:
    metrics = base_failure_metrics(df, target_col)
    insights = []

    if np.isfinite(metrics["failure_rate"]):
        insights.append(
            f"Failure events account for {metrics['failure_rate']:.2f}% of the observed machine states, "
            f"with {metrics['failures']:,} failures across {metrics['records']:,} records."
        )

    if not shift_table.empty:
        top = shift_table.iloc[0]
        insights.append(
            f"{top['feature']} shows the steepest risk gradient. Failure rate rises from "
            f"{top['low_range_failure_rate_pct']:.2f}% in lower operating ranges to "
            f"{top['high_range_failure_rate_pct']:.2f}% in higher ranges."
        )

        for _, row in shift_table.head(3).iterrows():
            insights.append(
                f"{row['feature']} is associated with a {row['change_pp']:.2f} percentage-point increase "
                f"in failure rate between lower and higher operating ranges."
            )

    if not median_table.empty:
        row = median_table.iloc[0]
        direction = "higher" if row["median_delta"] >= 0 else "lower"
        insights.append(
            f"Among failed machines, the median value for {row['feature']} is {abs(row['relative_change_pct']):.1f}% "
            f"{direction} than the non-failure group."
        )

    if not cat_table.empty:
        cat_row = cat_table.iloc[0]
        insights.append(
            f"The highest observed categorical risk concentration is in {cat_row['feature']} = {cat_row['category']}, "
            f"where the failure rate reaches {cat_row['failure_rate_pct']:.2f}% across {cat_row['records']:,} records."
        )

    return insights


# =========================
# Model helpers
# =========================
def build_pipeline(model_name: str, n_estimators: int, max_depth: Optional[int], random_state: int) -> Pipeline:
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

    prep = ColumnTransformer(
        transformers=[
            ("num", num_pipe, selector(dtype_include=np.number)),
            ("cat", cat_pipe, selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=int(random_state))
    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            class_weight="balanced",
            n_jobs=-1,
            random_state=int(random_state),
        )
    elif model_name == "Extra Trees":
        model = ExtraTreesClassifier(
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            class_weight="balanced",
            n_jobs=-1,
            random_state=int(random_state),
        )
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=int(random_state))
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(steps=[("prep", prep), ("model", model)])


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
    d = engineer_features(df.copy()).dropna(axis=0, how="all")
    d[target_col] = prepare_target_series(d, target_col)

    drop_cols = columns_to_drop_for_model(d, target_col)
    X = d.drop(columns=[target_col] + drop_cols, errors="ignore")
    y = d[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y if y.nunique() == 2 else None,
    )

    pipe = build_pipeline(model_name, n_estimators, max_depth, random_state)
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
        fpr, tpr, roc_thr = roc_curve(y_test, proba)
        precision_vals, recall_vals, pr_thr = precision_recall_curve(y_test, proba)
        roc = (fpr, tpr, roc_thr)
        pr = (precision_vals, recall_vals, pr_thr)
        ap = average_precision_score(y_test, proba)

    fi = pd.DataFrame(columns=["feature", "importance"])
    try:
        feature_names = [clean_feature_label(name) for name in pipe.named_steps["prep"].get_feature_names_out()]
        model_obj = pipe.named_steps["model"]

        if hasattr(model_obj, "feature_importances_"):
            fi = (
                pd.DataFrame(
                    {"feature": feature_names, "importance": model_obj.feature_importances_}
                )
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
        elif hasattr(model_obj, "coef_"):
            fi = (
                pd.DataFrame(
                    {"feature": feature_names, "importance": np.abs(model_obj.coef_[0])}
                )
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
    except Exception:
        pass

    error_table = pd.DataFrame()
    if proba is not None:
        review = X_test.copy()
        review["actual"] = y_test.to_numpy()
        review["predicted"] = pred
        review["failure_probability"] = proba
        review["absolute_error"] = np.abs(review["actual"] - review["failure_probability"])
        error_table = review.sort_values("absolute_error", ascending=False).reset_index(drop=True)

    return pipe, X, X_test, y_test, pred, proba, roc_auc, report, cm, fi, roc, pr, ap, error_table


@st.cache_data(show_spinner=False)
def compare_models_cached(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
    n_estimators: int,
    max_depth: Optional[int],
) -> pd.DataFrame:
    models = ["Gradient Boosting", "Random Forest", "Extra Trees", "Logistic Regression"]
    rows: list[dict] = []

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

    return pd.DataFrame(rows).sort_values(["F1", "ROC-AUC"], ascending=[False, False]).reset_index(drop=True)


def threshold_table(y_true: pd.Series, proba: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []
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


def model_driven_insights(
    compare_df: pd.DataFrame,
    report: dict,
    roc_auc: float,
    ap: float,
    fi: pd.DataFrame,
    selected_model: str,
) -> list[str]:
    insights = []

    if not compare_df.empty:
        best = compare_df.iloc[0]
        if best["Model"] == selected_model:
            insights.append(
                f"{selected_model} is currently the strongest option in this app configuration, leading the comparison on F1."
            )
        else:
            insights.append(
                f"{selected_model} is not the strongest option under the current settings. "
                f"{best['Model']} is leading on F1 and should be the default choice for failure detection."
            )

    accuracy = float(report.get("accuracy", np.nan))
    fail_prec = float(report.get("1", {}).get("precision", np.nan))
    fail_rec = float(report.get("1", {}).get("recall", np.nan))
    fail_f1 = float(report.get("1", {}).get("f1-score", np.nan))

    if np.isfinite(accuracy):
        insights.append(f"The selected model achieves {accuracy:.3f} accuracy on the holdout set.")
    if np.isfinite(roc_auc):
        insights.append(f"ROC-AUC is {roc_auc:.3f}, indicating the model can rank higher-risk machine states effectively.")
    if np.isfinite(ap):
        insights.append(
            f"Average Precision is {ap:.3f}, which is especially relevant when failure cases are rarer than normal operation."
        )
    if np.isfinite(fail_f1):
        insights.append(
            f"Failure-class performance is F1 {fail_f1:.3f}, with precision {fail_prec:.3f} and recall {fail_rec:.3f}."
        )
    if fi is not None and not fi.empty:
        top_features = ", ".join(fi.head(5)["feature"].tolist())
        insights.append(f"The strongest model drivers are {top_features}.")

    return insights


def threshold_commentary(thr_df: pd.DataFrame) -> list[str]:
    if thr_df.empty:
        return []

    best_f1 = thr_df.loc[thr_df["F1"].idxmax()]
    low_thr = thr_df.loc[thr_df["Threshold"].idxmin()]
    high_thr = thr_df.loc[thr_df["Threshold"].idxmax()]

    return [
        f"F1 peaks at a threshold of {best_f1['Threshold']:.2f}, balancing precision at {best_f1['Precision']:.3f} and recall at {best_f1['Recall']:.3f}.",
        f"At the lower end of the threshold range, recall reaches {low_thr['Recall']:.3f}, which supports earlier warning at the cost of more false positives.",
        f"At the higher end of the threshold range, precision improves to {high_thr['Precision']:.3f}, which is useful when maintenance capacity is constrained.",
    ]


# =========================
# Risk tool helpers
# =========================
def risk_band(probability: float, threshold: float) -> str:
    if probability < threshold * 0.7:
        return "Low risk"
    if probability < threshold:
        return "Medium risk"
    return "High risk"


def recommended_action(probability: float, threshold: float) -> str:
    band = risk_band(probability, threshold)
    if band == "High risk":
        return "Prioritise inspection or maintenance intervention soon."
    if band == "Medium risk":
        return "Monitor closely and schedule a near-term review if capacity allows."
    return "No immediate intervention is suggested under the current threshold."


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
        An insight-led dashboard for understanding failure patterns, comparing model behaviour,
        and translating predicted risk into practical maintenance action.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")


# =========================
# Sidebar
# =========================
st.sidebar.title("Configuration")

page = st.sidebar.radio(
    "View",
    [
        "Executive Overview",
        "Risk Drivers",
        "Model Review",
        "Scenario Lab",
        "Recommendations",
    ],
    index=0,
)

st.sidebar.divider()
st.sidebar.subheader("Data")

use_repo_data = st.sidebar.checkbox("Load data.csv from app folder", value=True)
if use_repo_data:
    if not DATA_PATH.exists():
        st.error("data.csv was not found in the same folder as app.py.")
        st.stop()
    df = load_csv_with_fallback(DATA_PATH)
    source_label = "Loaded from local data.csv"
else:
    upload = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if upload is None:
        st.info("Upload a CSV file or enable local data.csv loading.")
        st.stop()
    df = load_uploaded_csv(upload)
    source_label = "Loaded from uploaded CSV"

target_default = detect_target(df)
target_col = st.sidebar.selectbox(
    "Target column",
    options=df.columns.tolist(),
    index=df.columns.tolist().index(target_default),
)

st.sidebar.divider()
st.sidebar.subheader("Model settings")

model_name = st.sidebar.selectbox(
    "Selected model",
    ["Gradient Boosting", "Random Forest", "Extra Trees", "Logistic Regression"],
    index=0,
)
test_size = st.sidebar.slider("Test split", 0.10, 0.40, 0.20, step=0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=10_000, value=42)
n_estimators = st.sidebar.slider("Tree estimators", 50, 600, 300, step=50)
depth_choice = st.sidebar.selectbox("Max depth", ["None", "10", "20", "30"], index=0)
max_depth = None if depth_choice == "None" else int(depth_choice)
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.45, step=0.05)

st.sidebar.caption(source_label)

# =========================
# Shared analysis
# =========================
drop_cols = columns_to_drop_for_model(df, target_col)
shift_table = build_failure_shift_table(df, target_col, drop_cols)
median_table = build_median_comparison_table(df, target_col, drop_cols)
cat_table = categorical_failure_rates(df, target_col)
dataset_metrics = base_failure_metrics(df, target_col)
data_insights = generate_data_insights(df, target_col, shift_table, cat_table, median_table)

with st.spinner("Refreshing model outputs..."):
    compare_df = compare_models_cached(
        df=df,
        target_col=target_col,
        test_size=float(test_size),
        random_state=int(random_state),
        n_estimators=int(n_estimators),
        max_depth=max_depth,
    )
    (
        pipe,
        X_all,
        X_test,
        y_test,
        pred,
        proba,
        roc_auc,
        report,
        cm,
        fi,
        roc,
        pr,
        ap,
        error_table,
    ) = train_eval_cached(
        df=df,
        target_col=target_col,
        model_name=model_name,
        test_size=float(test_size),
        random_state=int(random_state),
        n_estimators=int(n_estimators),
        max_depth=max_depth,
    )

model_insights = model_driven_insights(compare_df, report, roc_auc, ap, fi, model_name)
best_model = compare_df.iloc[0]["Model"] if not compare_df.empty else "—"
selected_f1 = float(report.get("1", {}).get("f1-score", np.nan))
selected_recall = float(report.get("1", {}).get("recall", np.nan))
selected_precision = float(report.get("1", {}).get("precision", np.nan))


# =========================
# Page: Executive Overview
# =========================
if page == "Executive Overview":
    k1, k2, k3, k4 = st.columns(4, gap="large")
    with k1:
        kpi_card(
            "Observed failure rate",
            f"{dataset_metrics['failure_rate']:.2f}%" if np.isfinite(dataset_metrics["failure_rate"]) else "—",
            "Share of machine states labelled as failure",
        )
    with k2:
        kpi_card(
            "Failure cases",
            f"{dataset_metrics['failures']:,}",
            "Observed failure events in the dataset",
        )
    with k3:
        leading_driver = shift_table.iloc[0]["feature"] if not shift_table.empty else "Not available"
        kpi_card(
            "Strongest risk driver",
            leading_driver,
            "Largest increase in failure rate across operating ranges",
        )
    with k4:
        kpi_card(
            "Best model by F1",
            best_model,
            "Highest holdout failure-detection balance",
        )

    st.write("")

    left, right = st.columns([1.18, 0.82], gap="large")

    with left:
        section_card_start(
            "Executive Summary",
            "The first read on what matters operationally in the current maintenance dataset",
        )
        st.markdown("**Data-driven insights**")
        display_insights(data_insights, limit=4)
        st.write("")
        st.markdown("**Model-driven insights**")
        display_insights(model_insights, limit=4)
        section_card_end()

    with right:
        section_card_start(
            "Decision Support Snapshot",
            "How the current model configuration translates into maintenance prioritisation quality",
        )
        snapshot_cols = st.columns(2)
        with snapshot_cols[0]:
            kpi_card(
                "Failure F1",
                f"{selected_f1:.3f}" if np.isfinite(selected_f1) else "—",
                "Balance between precision and recall on failures",
            )
        with snapshot_cols[1]:
            kpi_card(
                "ROC-AUC",
                f"{roc_auc:.3f}" if np.isfinite(roc_auc) else "—",
                "Ability to rank higher-risk machine states",
            )
        st.write("")
        if not compare_df.empty:
            top_compare = compare_df[["Model", "F1", "ROC-AUC"]].copy()
            fig_compare = px.bar(
                top_compare.sort_values("F1"),
                x="F1",
                y="Model",
                orientation="h",
                text="F1",
            )
            fig_compare.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_compare.update_layout(height=290, margin=dict(l=0, r=15, t=10, b=0))
            st.plotly_chart(fig_compare, use_container_width=True)
        section_card_end()

    st.write("")

    a, b = st.columns([1.0, 1.0], gap="large")

    with a:
        section_card_start(
            "Which operating signals matter most",
            "Failure-rate shift from lower to higher operating ranges",
        )
        if shift_table.empty:
            st.info("Not enough numeric signal variation was detected to calculate failure-rate gradients.")
        else:
            show = shift_table.head(10).iloc[::-1]
            fig_shift = px.bar(
                show,
                x="change_pp",
                y="feature",
                orientation="h",
                labels={"change_pp": "Increase in failure rate (percentage points)", "feature": "Feature"},
            )
            fig_shift.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_shift, use_container_width=True)
        section_card_end()

    with b:
        section_card_start(
            "Where the selected model creates the most value",
            "Threshold-aware view of failure detection quality",
        )
        if proba is None:
            st.info("Probability-based threshold analysis is not available for the current model.")
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
            for note in threshold_commentary(thr_df):
                st.markdown(f'<div class="small" style="margin-top:6px;">{note}</div>', unsafe_allow_html=True)
        section_card_end()


# =========================
# Page: Risk Drivers
# =========================
elif page == "Risk Drivers":
    st.markdown("### Risk Drivers")
    st.caption("This section explains what the operating data is saying before moving into model behaviour.")

    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        kpi_card("Total records", f"{dataset_metrics['records']:,}", "Machine observations available for analysis")
    with c2:
        kpi_card("Failure events", f"{dataset_metrics['failures']:,}", "Observed failure cases")
    with c3:
        risk_gap = shift_table.iloc[0]["change_pp"] if not shift_table.empty else np.nan
        kpi_card("Largest risk gradient", f"{risk_gap:.2f} pp" if np.isfinite(risk_gap) else "—", "Difference between lower and higher ranges")
    with c4:
        top_cat = cat_table.iloc[0]["category"] if not cat_table.empty else "Not available"
        kpi_card("Highest-risk category", top_cat, "Category with the highest observed failure rate")

    st.write("")

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        section_card_start(
            "Data-driven insights",
            "Plain-language findings derived directly from the observed machine data",
        )
        display_insights(data_insights, limit=8)
        section_card_end()

    with right:
        section_card_start(
            "Highest-risk categorical segments",
            "Useful when prioritising interventions by machine group or type",
        )
        if cat_table.empty:
            st.info("No categorical segment breakdown produced meaningful failure-rate differences.")
        else:
            fig_cat = px.bar(
                cat_table.iloc[::-1],
                x="failure_rate_pct",
                y="category",
                color="feature",
                orientation="h",
                labels={"failure_rate_pct": "Failure rate (%)", "category": "Category"},
            )
            fig_cat.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_cat, use_container_width=True)
            st.dataframe(cat_table, use_container_width=True, hide_index=True)
        section_card_end()

    st.write("")

    a, b = st.columns([1.0, 1.0], gap="large")

    with a:
        section_card_start(
            "Risk gradient by feature",
            "Features ranked by how sharply failure rate rises across operating ranges",
        )
        if shift_table.empty:
            st.info("Risk gradients could not be calculated for the current dataset.")
        else:
            top_n = st.slider("Number of features to display", 5, min(20, len(shift_table)), 10)
            show = shift_table.head(top_n).iloc[::-1]
            fig_shift = px.bar(
                show,
                x="change_pp",
                y="feature",
                orientation="h",
                labels={"change_pp": "Increase in failure rate (percentage points)", "feature": "Feature"},
            )
            fig_shift.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_shift, use_container_width=True)
        section_card_end()

    with b:
        section_card_start(
            "How failure rate changes across a selected operating signal",
            "This view shows the direction and shape of risk rather than the raw feature distribution",
        )
        if shift_table.empty:
            st.info("No numeric operating signal is available for this view.")
        else:
            selected_feature = st.selectbox(
                "Operating signal",
                options=shift_table["feature"].tolist(),
                index=0,
            )
            stats = compute_numeric_group_stats(engineer_features(df.copy()), target_col, selected_feature)
            if not stats:
                st.info("Not enough valid observations exist for this feature.")
            else:
                trend = stats["table"].copy()
                fig_trend = px.line(
                    trend,
                    x="feature_median",
                    y="failure_rate",
                    markers=True,
                    labels={"feature_median": selected_feature, "failure_rate": "Failure rate"},
                )
                fig_trend.update_yaxes(tickformat=".0%")
                fig_trend.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_trend, use_container_width=True)
                st.markdown(
                    f'<div class="small">{selected_feature} moves from approximately '
                    f'{stats["low_failure_rate_pct"]:.2f}% failure rate in lower ranges to '
                    f'{stats["high_failure_rate_pct"]:.2f}% in higher ranges.</div>',
                    unsafe_allow_html=True,
                )
        section_card_end()

    st.write("")

    section_card_start(
        "Median operating profile: failure vs non-failure states",
        "This comparison highlights where failed machines diverge most from normal operation",
    )
    if median_table.empty:
        st.info("Median comparison could not be calculated for the current dataset.")
    else:
        show_medians = median_table.head(12).copy()
        fig_median = go.Figure()
        fig_median.add_trace(
            go.Bar(
                x=show_medians["feature"],
                y=show_medians["non_failure_median"],
                name="Non-failure median",
            )
        )
        fig_median.add_trace(
            go.Bar(
                x=show_medians["feature"],
                y=show_medians["failure_median"],
                name="Failure median",
            )
        )
        fig_median.update_layout(
            barmode="group",
            height=420,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Feature",
            yaxis_title="Median value",
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_median, use_container_width=True)
        st.dataframe(show_medians, use_container_width=True, hide_index=True)
    section_card_end()


# =========================
# Page: Model Review
# =========================
elif page == "Model Review":
    st.markdown("### Model Review")
    st.caption("This section evaluates how reliably the selected model identifies failure risk and how it compares with alternatives.")

    m1, m2, m3, m4 = st.columns(4, gap="large")
    with m1:
        kpi_card("Failure precision", f"{selected_precision:.3f}" if np.isfinite(selected_precision) else "—", "Share of predicted failures that were correct")
    with m2:
        kpi_card("Failure recall", f"{selected_recall:.3f}" if np.isfinite(selected_recall) else "—", "Share of actual failures captured")
    with m3:
        kpi_card("Failure F1", f"{selected_f1:.3f}" if np.isfinite(selected_f1) else "—", "Balanced failure-detection score")
    with m4:
        kpi_card("Average Precision", f"{ap:.3f}" if np.isfinite(ap) else "—", "Precision-recall summary across thresholds")

    st.write("")

    left, right = st.columns([1.0, 1.0], gap="large")

    with left:
        section_card_start(
            "Model-driven insights",
            "Interpretation of what the current evaluation says about operational usefulness",
        )
        display_insights(model_insights, limit=8)
        section_card_end()

    with right:
        section_card_start(
            "Model comparison",
            "The selected model should justify itself against alternatives rather than stand alone",
        )
        st.dataframe(compare_df, use_container_width=True, hide_index=True)
        section_card_end()

    st.write("")

    a, b = st.columns([1.0, 1.0], gap="large")

    with a:
        section_card_start(
            "Confusion matrix",
            "How predictions split between correct alerts, missed failures, and false alarms",
        )
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: No failure", "Actual: Failure"],
            columns=["Predicted: No failure", "Predicted: Failure"],
        )
        fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto")
        fig_cm.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_cm, use_container_width=True)
        section_card_end()

    with b:
        section_card_start(
            "Threshold trade-off",
            "Adjusting the alert threshold changes the balance between missed failures and unnecessary inspections",
        )
        if proba is None:
            st.info("Threshold analysis is not available without predicted probabilities.")
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
                xaxis_title="Threshold",
                yaxis_title="Score",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_thr, use_container_width=True)
            best_row = thr_df.loc[thr_df["F1"].idxmax()]
            st.markdown(
                f'<div class="small">Best F1 is reached near threshold {best_row["Threshold"]:.2f} '
                f'with precision {best_row["Precision"]:.3f} and recall {best_row["Recall"]:.3f}.</div>',
                unsafe_allow_html=True,
            )
        section_card_end()

    st.write("")

    c, d = st.columns([1.0, 1.0], gap="large")

    with c:
        section_card_start(
            "Ranking quality",
            "These curves show how well the model separates higher-risk from lower-risk machine states",
        )
        if roc[0] is None or pr[0] is None:
            st.info("Ranking diagnostics are not available for the current model.")
        else:
            fpr, tpr, _ = roc
            precision_vals, recall_vals, _ = pr

            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline"))
            fig_roc.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="False positive rate",
                yaxis_title="True positive rate",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=recall_vals, y=precision_vals, mode="lines", name="Precision-Recall"))
            fig_pr.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Recall",
                yaxis_title="Precision",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_pr, use_container_width=True)
        section_card_end()

    with d:
        section_card_start(
            "Main model drivers",
            "The strongest drivers help explain what the selected model pays attention to",
        )
        if fi.empty:
            st.info("Feature importance is not available for the current model.")
        else:
            show_top = st.slider("Top drivers to display", 5, 25, 12)
            fi_top = fi.head(show_top).iloc[::-1]
            fig_fi = px.bar(fi_top, x="importance", y="feature", orientation="h")
            fig_fi.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_fi, use_container_width=True)
        section_card_end()

    st.write("")

    section_card_start(
        "Cases with the greatest prediction uncertainty",
        "These records are the best candidates for manual review, rule-based checks, or future error analysis",
    )
    if error_table.empty:
        st.info("Prediction error review is not available for the current model.")
    else:
        review_cols = [col for col in error_table.columns if col not in {"actual", "predicted"}][:6] + [
            "actual",
            "predicted",
            "failure_probability",
            "absolute_error",
        ]
        review_cols = [col for col in review_cols if col in error_table.columns]
        st.dataframe(error_table[review_cols].head(15), use_container_width=True, hide_index=True)
    section_card_end()


# =========================
# Page: Scenario Lab
# =========================
elif page == "Scenario Lab":
    st.markdown("### Scenario Lab")
    st.caption("This tool turns the trained model into a practical risk-testing workflow for maintenance decisions.")

    df_pred = engineer_features(df.copy())
    model_drop_cols = columns_to_drop_for_model(df_pred, target_col)
    X = df_pred.drop(columns=[target_col] + model_drop_cols, errors="ignore").copy()

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        kpi_card("Selected model", model_name, "Current prediction engine")
    with c2:
        kpi_card("Decision threshold", f"{threshold:.2f}", "Alert cut-off for failure classification")
    with c3:
        typical_profile = shift_table.iloc[0]["feature"] if not shift_table.empty else "Not available"
        kpi_card("Primary monitoring signal", typical_profile, "Highest-risk signal under current data")

    st.write("")

    section_card_start(
        "Scenario inputs",
        "Use realistic machine conditions to test how risk changes under different operating states",
    )

    input_row: dict[str, object] = {}
    cols = st.columns(3, gap="large")

    for idx, col in enumerate(X.columns):
        holder = cols[idx % 3]
        series = X[col]

        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.empty:
                input_row[col] = 0.0
            else:
                lower = float(numeric.quantile(0.05))
                upper = float(numeric.quantile(0.95))
                median = float(numeric.median())
                if np.isclose(lower, upper):
                    input_row[col] = holder.number_input(col, value=median, step=0.1)
                else:
                    input_row[col] = holder.slider(col, min_value=lower, max_value=upper, value=median)
        else:
            values = sorted(series.dropna().astype(str).unique().tolist())
            input_row[col] = holder.selectbox(col, values) if values else ""

    section_card_end()

    X_new = pd.DataFrame([input_row])

    if hasattr(pipe.named_steps["model"], "predict_proba"):
        probability = float(pipe.predict_proba(X_new)[:, 1][0])
    else:
        probability = float(pipe.predict(X_new)[0])

    decision = 1 if probability >= threshold else 0
    band = risk_band(probability, threshold)
    action = recommended_action(probability, threshold)
    band_color = {"Low risk": SUCCESS, "Medium risk": WARNING, "High risk": DANGER}[band]

    left, right = st.columns([0.82, 1.18], gap="large")

    with left:
        section_card_start(
            "Scenario result",
            "This summary translates the predicted score into an operational recommendation",
        )
        st.markdown(
            f"""
            <div class="callout">
                <div class="metric-title">Predicted failure probability</div>
                <div class="metric-value">{probability:.2%}</div>
                <div class="metric-sub">Predicted class: {"Failure" if decision == 1 else "No failure"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            f"<div style='font-weight:800; font-size:18px; color:{band_color};'>Risk band: {band}</div>",
            unsafe_allow_html=True,
        )
        st.write(f"Recommended action: **{action}**")
        st.markdown(
            f'<div class="small">The risk band uses the current threshold of {threshold:.2f}. '
            f'Lower thresholds create earlier alerts, while higher thresholds reduce unnecessary interventions.</div>',
            unsafe_allow_html=True,
        )
        section_card_end()

    with right:
        section_card_start(
            "Failure probability gauge",
            "A fast visual read on whether the current scenario is below, near, or above the intervention threshold",
        )
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%"},
                title={"text": "Failure probability"},
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
        gauge.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(gauge, use_container_width=True)
        section_card_end()

    st.write("")

    section_card_start(
        "Input profile vs typical operating level",
        "This view shows where the scenario sits relative to the median operating state in the historical data",
    )
    numeric_comparison = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            median_val = float(pd.to_numeric(X[col], errors="coerce").dropna().median())
            scenario_val = float(input_row[col])
            numeric_comparison.append(
                {
                    "feature": col,
                    "scenario_value": scenario_val,
                    "historical_median": median_val,
                    "delta": scenario_val - median_val,
                }
            )
    compare_inputs = pd.DataFrame(numeric_comparison)
    if compare_inputs.empty:
        st.info("No numeric fields are available for profile comparison.")
    else:
        show_inputs = compare_inputs.copy().sort_values("delta", key=np.abs, ascending=False).head(12)
        fig_inputs = go.Figure()
        fig_inputs.add_trace(go.Bar(x=show_inputs["feature"], y=show_inputs["historical_median"], name="Historical median"))
        fig_inputs.add_trace(go.Bar(x=show_inputs["feature"], y=show_inputs["scenario_value"], name="Scenario"))
        fig_inputs.update_layout(
            barmode="group",
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Feature",
            yaxis_title="Value",
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_inputs, use_container_width=True)
        st.dataframe(show_inputs, use_container_width=True, hide_index=True)
    section_card_end()


# =========================
# Page: Recommendations
# =========================
elif page == "Recommendations":
    st.markdown("### Recommendations")
    st.caption("This final section turns the analysis into a concise portfolio-ready operational story.")

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        kpi_card("Recommended default model", best_model, "Highest F1 under the current evaluation setup")
    with c2:
        best_threshold_df = threshold_table(y_test, proba) if proba is not None else pd.DataFrame()
        best_threshold = best_threshold_df.loc[best_threshold_df["F1"].idxmax(), "Threshold"] if not best_threshold_df.empty else np.nan
        kpi_card("Best F1 threshold", f"{best_threshold:.2f}" if np.isfinite(best_threshold) else "—", "Strongest balance of precision and recall")
    with c3:
        top_driver = fi.iloc[0]["feature"] if not fi.empty else "Not available"
        kpi_card("Priority monitoring signal", top_driver, "Most important signal in the current model")

    st.write("")

    left, right = st.columns([1.0, 1.0], gap="large")

    with left:
        section_card_start(
            "Operational recommendations",
            "How this dashboard can support inspection planning and earlier intervention",
        )
        display_insights(
            [
                "Use the model as a ranking tool, not just a binary classifier, so maintenance teams can prioritise the highest-risk machine states first.",
                "Adopt a lower alert threshold when the cost of missed failure is high and maintenance resources can absorb more inspections.",
                "Track the strongest risk signals routinely because they show where failure rates accelerate most sharply.",
                "Review the highest-uncertainty cases manually to identify rule gaps, labelling issues, or patterns the model still misses.",
                "Retrain the model periodically with fresh operating data so the decision logic stays aligned with current machine behaviour.",
            ]
        )
        section_card_end()

    with right:
        section_card_start(
            "Portfolio positioning",
            "How to explain the value of this project in interviews or portfolio reviews",
        )
        display_insights(
            [
                "This project goes beyond a standard classifier by combining risk analysis, model comparison, threshold tuning, and interactive scenario testing.",
                "The dashboard is designed for decision support, helping stakeholders understand which operating conditions raise risk and how alert strategy changes outcomes.",
                "The model review makes the chosen approach defendable because it shows why one model is stronger than the alternatives under the same holdout conditions.",
                "The scenario lab turns the project into a practical tool that supports maintenance prioritisation rather than ending at an evaluation table.",
            ]
        )
        section_card_end()

    st.write("")

    section_card_start(
        "Suggested next upgrades",
        "These additions would make the project even stronger for real-world deployment",
    )
    display_insights(
        [
            "Add SHAP-based explanations to show global drivers and local record-level justification for individual risk scores.",
            "Incorporate cost-sensitive evaluation so threshold decisions can reflect downtime costs, inspection cost, and the penalty of missed failures.",
            "Compare performance by machine type or operating regime to detect whether one model underperforms on specific subgroups.",
            "Persist the strongest model and expose a production-style scoring workflow for batch or API-based risk scoring.",
            "Add drift monitoring so changes in operating patterns can trigger retraining or investigation before performance degrades.",
        ]
    )
    section_card_end()