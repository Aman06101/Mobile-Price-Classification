"""Streamlit frontend for Aman's Mobile Price Classification project."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "model"
MODEL_CARD_PATH = MODEL_ROOT / "model_card.json"
COMPARISON_PATH = MODEL_ROOT / "comparison.csv"
SAVED_MODEL_ROOT = MODEL_ROOT / "saved_models"
LABELLED_SAMPLE_PATH = ROOT / "test_data.csv"
UNLABELLED_SAMPLE_PATH = ROOT / "data" / "mobile_price_unlabelled.csv"
TARGET = "price_range"
ID_COLUMN = "id"

st.set_page_config(
    page_title="Mobile Price Band Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #fff8ef 0%, #ffffff 45%, #f3f1ff 100%);
    }
    .mp-hero {
        padding: 1.25rem 1.4rem;
        border-radius: 18px;
        background: #24144f;
        color: white;
        border-bottom: 5px solid #ff7a00;
        margin-bottom: 1rem;
    }
    .mp-hero h1 { margin: 0; font-size: 2.15rem; }
    .mp-hero p { margin: .35rem 0 0 0; opacity: .92; }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.92);
        border: 1px solid #eadff8;
        border-radius: 12px;
        padding: .7rem;
    }
    .price-note {
        background: #fff3df;
        border-left: 5px solid #ff7a00;
        padding: .75rem 1rem;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_json(path_text: str) -> dict:
    return json.loads(Path(path_text).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv(path_text: str) -> pd.DataFrame:
    return pd.read_csv(path_text)


@st.cache_resource(show_spinner=False)
def load_pipeline(path_text: str):
    return joblib.load(path_text)


def score_text(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.4f}"


def calculate_scores(
    y_true: pd.Series,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float | None]:
    auc_value: float | None
    try:
        auc_value = float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=classes,
                multi_class="ovr",
                average="weighted",
            )
        )
    except ValueError:
        auc_value = None

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": auc_value,
        "Precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "Recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def show_metric_cards(scores: dict[str, float | None]) -> None:
    first = st.columns(3)
    second = st.columns(3)
    for box, metric_name in zip(
        first + second,
        ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
    ):
        box.metric(metric_name, score_text(scores[metric_name]))


def prepare_frame(
    raw: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None, list[str]]:
    if raw.empty:
        raise ValueError("The CSV is empty.")

    frame = raw.copy()
    frame.columns = [str(column).strip() for column in frame.columns]

    identifiers = frame[ID_COLUMN].copy() if ID_COLUMN in frame.columns else None
    target = None
    if TARGET in frame.columns:
        numeric_target = pd.to_numeric(frame[TARGET], errors="coerce")
        if numeric_target.isna().any():
            raise ValueError("The price_range column must contain only 0, 1, 2, or 3.")
        numeric_target = numeric_target.astype(int)
        unexpected = sorted(set(numeric_target.unique()) - {0, 1, 2, 3})
        if unexpected:
            raise ValueError(f"Unexpected price_range values: {unexpected}")
        target = numeric_target

    missing = [column for column in feature_names if column not in frame.columns]
    if missing:
        raise ValueError(
            "Missing required feature columns: " + ", ".join(missing)
        )

    features = frame[feature_names].copy()
    converted_with_missing: list[str] = []
    for column in feature_names:
        before_missing = int(features[column].isna().sum())
        features[column] = pd.to_numeric(features[column], errors="coerce")
        after_missing = int(features[column].isna().sum())
        if after_missing > before_missing:
            converted_with_missing.append(column)

    return features, target, identifiers, converted_with_missing


def build_prediction_table(
    original: pd.DataFrame,
    identifiers: pd.Series | None,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    class_names: dict[int, str],
) -> pd.DataFrame:
    output = pd.DataFrame(index=original.index)
    if identifiers is not None:
        output[ID_COLUMN] = identifiers.to_numpy()
    output["predicted_price_range"] = predictions
    output["predicted_price_label"] = [class_names[int(value)] for value in predictions]
    for index, class_value in enumerate(classes):
        output[f"probability_class_{int(class_value)}"] = probabilities[:, index]
    if TARGET in original.columns:
        output["actual_price_range"] = pd.to_numeric(
            original[TARGET], errors="coerce"
        ).astype("Int64")
    return output


def render_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(6.1, 4.6))
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        display_labels=["Budget", "Lower Mid", "Upper Mid", "Premium"],
        cmap="Purples",
        colorbar=False,
        values_format="d",
        ax=axis,
    )
    display.ax_.set_title("Confusion matrix")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=False)
    plt.close(figure)


try:
    card = load_json(str(MODEL_CARD_PATH))
    comparison = load_csv(str(COMPARISON_PATH))
except Exception as exc:
    st.error(
        "Required model files are missing. Run `python -m model.mobile_training` "
        "from the repository root, then restart the app."
    )
    st.exception(exc)
    st.stop()

feature_names = list(card["dataset"]["features"])
class_names = {int(key): value for key, value in card["dataset"]["class_labels"].items()}
model_names = list(card["models"]["order"])

st.markdown(
    f"""
    <div class="mp-hero">
        <h1>Mobile Price Band Predictor</h1>
        <p>Four-class classification for mobile phone price tiers</p>
        <p><strong>{card['student']['name']}</strong> · BITS ID {card['student']['bits_id']}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([2, 1])
with left:
    selected_model = st.selectbox("Choose a classification model", model_names)
with right:
    st.metric("Labelled test rows", card["split"]["test_rows"])

view = st.radio(
    "Choose what to do",
    ["Compare models", "Evaluate labelled CSV", "Predict unlabelled CSV"],
    horizontal=True,
)

model_path = SAVED_MODEL_ROOT / card["models"]["files"][selected_model]
model = load_pipeline(str(model_path))
selected_saved_scores = (
    comparison.loc[comparison["Model"] == selected_model]
    .iloc[0]
    .drop(labels="Model")
    .to_dict()
)

if view == "Compare models":
    st.subheader(f"Saved test results: {selected_model}")
    show_metric_cards({key: float(value) for key, value in selected_saved_scores.items()})

    st.subheader("All model results")
    displayed = comparison.copy()
    for metric in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
        displayed[metric] = displayed[metric].map(lambda value: f"{value:.4f}")
    st.dataframe(displayed, hide_index=True, use_container_width=True)

    winner = comparison.sort_values(["MCC", "F1", "AUC"], ascending=False).iloc[0]
    st.markdown(
        f"""
        <div class="price-note">
        <strong>Overall winner:</strong> {winner['Model']} has the strongest combined
        MCC, weighted F1, and AUC on the fixed test split. Weighted averaging is used
        because this is a four-class problem.
        </div>
        """,
        unsafe_allow_html=True,
    )

elif view == "Evaluate labelled CSV":
    st.subheader("Evaluate a CSV containing price_range")
    source = st.radio(
        "Data source",
        ["Use bundled test_data.csv", "Upload my CSV"],
        horizontal=True,
    )
    if source == "Use bundled test_data.csv":
        raw_data = load_csv(str(LABELLED_SAMPLE_PATH))
    else:
        uploaded = st.file_uploader("Upload labelled CSV", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV to continue.")
            st.stop()
        raw_data = pd.read_csv(uploaded)

    try:
        features, target, identifiers, conversion_warnings = prepare_frame(
            raw_data, feature_names
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if conversion_warnings:
        st.warning(
            "Non-numeric values were converted to missing values in: "
            + ", ".join(conversion_warnings)
            + ". The saved pipeline will impute them using training medians."
        )

    probabilities = model.predict_proba(features)
    predictions = model.predict(features)
    output = build_prediction_table(
        raw_data,
        identifiers,
        predictions,
        probabilities,
        model.classes_,
        class_names,
    )

    if target is None:
        st.warning(
            "The uploaded CSV has no price_range column, so evaluation metrics cannot "
            "be calculated. Predictions are still available below."
        )
    else:
        live_scores = calculate_scores(target, predictions, probabilities, model.classes_)
        show_metric_cards(live_scores)
        if live_scores["AUC"] is None:
            st.info("AUC is unavailable because the uploaded data does not include all four classes.")

        plot_column, report_column = st.columns([1, 1.15])
        with plot_column:
            render_confusion_matrix(target, predictions)
        with report_column:
            report = classification_report(
                target,
                predictions,
                labels=[0, 1, 2, 3],
                target_names=[class_names[index] for index in range(4)],
                output_dict=True,
                zero_division=0,
            )
            st.markdown("#### Classification report")
            st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

    st.markdown("#### Prediction preview")
    st.dataframe(output.head(50), hide_index=True, use_container_width=True)
    st.download_button(
        "Download predictions",
        data=output.to_csv(index=False).encode("utf-8"),
        file_name="mobile_price_predictions.csv",
        mime="text/csv",
    )

else:
    st.subheader("Predict the original unlabelled mobile test file")
    source = st.radio(
        "Data source",
        ["Use bundled Kaggle test file", "Upload another unlabelled CSV"],
        horizontal=True,
    )
    if source == "Use bundled Kaggle test file":
        raw_data = load_csv(str(UNLABELLED_SAMPLE_PATH))
    else:
        uploaded = st.file_uploader("Upload unlabelled CSV", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV to continue.")
            st.stop()
        raw_data = pd.read_csv(uploaded)

    try:
        features, target, identifiers, conversion_warnings = prepare_frame(
            raw_data, feature_names
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    probabilities = model.predict_proba(features)
    predictions = model.predict(features)
    output = build_prediction_table(
        raw_data,
        identifiers,
        predictions,
        probabilities,
        model.classes_,
        class_names,
    )

    if target is not None:
        st.info(
            "A price_range column was found. Use the labelled evaluation view when you "
            "want metrics and a confusion matrix."
        )
    else:
        st.caption(
            "The Kaggle test file has no price_range target, so this section produces "
            "predictions only."
        )

    counts = (
        output["predicted_price_label"]
        .value_counts()
        .rename_axis("Price tier")
        .reset_index(name="Predicted phones")
    )
    chart_column, table_column = st.columns([1, 1.4])
    with chart_column:
        st.bar_chart(counts.set_index("Price tier"))
    with table_column:
        st.dataframe(output.head(50), hide_index=True, use_container_width=True)

    st.download_button(
        "Download unlabelled predictions",
        data=output.to_csv(index=False).encode("utf-8"),
        file_name="mobile_unlabelled_predictions.csv",
        mime="text/csv",
    )

st.caption(
    "The app loads already-fitted scikit-learn pipelines. Uploaded files are used only "
    "for evaluation or prediction and are never added to model training."
)
