"""Train and persist five classifiers for the Mobile Price Classification dataset.

Run from the repository root:
    python -m model.mobile_training
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data" / "mobile_price_train.csv"
UNLABELLED_FILE = ROOT / "data" / "mobile_price_unlabelled.csv"
TEST_FILE = ROOT / "test_data.csv"
MODEL_DIR = ROOT / "model"
SAVED_MODEL_DIR = MODEL_DIR / "saved_models"
COMPARISON_FILE = MODEL_DIR / "comparison.csv"
MODEL_CARD_FILE = MODEL_DIR / "model_card.json"

TARGET = "price_range"
ID_COLUMN = "id"
RANDOM_STATE = 123
TEST_SIZE = 0.20

FEATURES = [
    "battery_power",
    "blue",
    "clock_speed",
    "dual_sim",
    "fc",
    "four_g",
    "int_memory",
    "m_dep",
    "mobile_wt",
    "n_cores",
    "pc",
    "px_height",
    "px_width",
    "ram",
    "sc_h",
    "sc_w",
    "talk_time",
    "three_g",
    "touch_screen",
    "wifi",
]

CLASS_LABELS = {
    0: "Budget",
    1: "Lower Mid-Range",
    2: "Upper Mid-Range",
    3: "Premium",
}

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}


def load_training_frame(path: Path = TRAIN_FILE) -> pd.DataFrame:
    """Load the labelled training data and enforce the expected schema."""
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    frame = pd.read_csv(path)
    missing = [column for column in FEATURES + [TARGET] if column not in frame.columns]
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")

    frame = frame[FEATURES + [TARGET]].drop_duplicates().copy()
    for column in FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise").astype(int)
    observed_classes = set(frame[TARGET].unique().tolist())
    expected_classes = set(CLASS_LABELS)
    if observed_classes != expected_classes:
        raise ValueError(
            f"Expected target classes {sorted(expected_classes)}, "
            f"but found {sorted(observed_classes)}"
        )
    return frame


def _scaled_pipeline(estimator: Any, scaler: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
            ("classifier", estimator),
        ]
    )


def _tree_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", estimator),
        ]
    )


def build_model_catalog() -> dict[str, Pipeline]:
    """Create independent preprocessing-and-model pipelines."""
    return {
        "Logistic Regression": _scaled_pipeline(
            LogisticRegression(
                C=3.0,
                max_iter=5000,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
            StandardScaler(),
        ),
        "Decision Tree": _tree_pipeline(
            DecisionTreeClassifier(
                criterion="entropy",
                max_depth=10,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
            )
        ),
        "kNN": _scaled_pipeline(
            KNeighborsClassifier(
                n_neighbors=31,
                weights="distance",
                metric="minkowski",
                p=2,
            ),
            RobustScaler(),
        ),
        "Naive Bayes": _scaled_pipeline(
            GaussianNB(var_smoothing=1e-9),
            StandardScaler(),
        ),
        "Random Forest": _tree_pipeline(
            RandomForestClassifier(
                n_estimators=350,
                max_depth=15,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
    }


def calculate_multiclass_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Calculate the six assignment metrics for four-class classification."""
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=sorted(CLASS_LABELS),
                multi_class="ovr",
                average="weighted",
            )
        ),
        "Precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "Recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def train_and_save() -> pd.DataFrame:
    """Train all models, export the untouched test split, and save artifacts."""
    frame = load_training_frame()
    X = frame[FEATURES]
    y = frame[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    test_export = X_test.copy()
    test_export[TARGET] = y_test.to_numpy()
    test_export.to_csv(TEST_FILE, index=False)

    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []

    for model_name, pipeline in build_model_catalog().items():
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        probability = pipeline.predict_proba(X_test)
        scores = calculate_multiclass_metrics(y_test, prediction, probability)
        rows.append({"Model": model_name, **scores})

        model_path = SAVED_MODEL_DIR / MODEL_FILES[model_name]
        joblib.dump(pipeline, model_path, compress=3)

        reloaded = joblib.load(model_path)
        if not np.array_equal(prediction, reloaded.predict(X_test)):
            raise RuntimeError(f"Reload verification failed for {model_name}")

    comparison = pd.DataFrame(rows)
    comparison.to_csv(COMPARISON_FILE, index=False)

    unlabelled_rows = None
    if UNLABELLED_FILE.exists():
        unlabelled_rows = int(pd.read_csv(UNLABELLED_FILE).shape[0])

    model_card = {
        "student": {
            "name": "Aman Singh",
            "bits_id": "2025AC05123",
        },
        "project": "Mobile Price Classification",
        "dataset_url": "https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification",
        "dataset": {
            "labelled_rows": int(frame.shape[0]),
            "unlabelled_rows": unlabelled_rows,
            "feature_count": len(FEATURES),
            "target": TARGET,
            "features": FEATURES,
            "class_labels": {str(key): value for key, value in CLASS_LABELS.items()},
        },
        "split": {
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
        },
        "metrics": {
            "multiclass_average": "weighted",
            "auc_strategy": "one-vs-rest weighted",
            "columns": ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
        },
        "models": {
            "order": list(MODEL_FILES),
            "files": MODEL_FILES,
        },
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    MODEL_CARD_FILE.write_text(json.dumps(model_card, indent=2), encoding="utf-8")
    return comparison


def main() -> None:
    comparison = train_and_save()
    print("\nMobile price models trained successfully.\n")
    print(comparison.round(4).to_string(index=False))
    print(f"\nSaved labelled test data: {TEST_FILE.relative_to(ROOT)}")
    print(f"Saved model artifacts: {SAVED_MODEL_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
