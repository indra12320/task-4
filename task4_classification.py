"""
MainCrafts AI & ML Internship - Task 4
Classification Models, Evaluation Metrics & Handling Imbalanced Data

Dataset: Breast Cancer Wisconsin dataset (built into scikit-learn)
Run: python task4_classification.py
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay, ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight
import joblib

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
REPORTS = BASE / "reports"
GRAPHS = REPORTS / "graphs"
ARTIFACTS = BASE / "artifacts"
for folder in (REPORTS, GRAPHS, ARTIFACTS):
    folder.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading Breast Cancer Wisconsin dataset...")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    # Dataset labels: 0=malignant, 1=benign. Make malignant the positive class (1).
    y = pd.Series((data.target == 0).astype(int), name="malignant")
    class_names = ["Benign (0)", "Malignant (1)"]

    print(f"Dataset rows: {len(X)} | features: {X.shape[1]}")
    print("Class counts (0=benign, 1=malignant):")
    counts = y.value_counts().sort_index()
    print(counts.to_string())

    # Split before fitting any preprocessing to avoid leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=42))
        ]),
        "Logistic Regression (Balanced)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Decision Tree (Balanced)": DecisionTreeClassifier(class_weight="balanced", random_state=42)
    }

    rows = []
    fitted = {}
    probabilities = {}
    predictions = {}

    print("\nTraining and evaluating classifiers...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
        else:
            proba = pred.astype(float)

        fitted[name] = model
        probabilities[name] = proba
        predictions[name] = pred
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "ROC AUC": roc_auc_score(y_test, proba),
        })

    metrics_df = pd.DataFrame(rows).sort_values("F1 Score", ascending=False)
    metrics_df.to_csv(REPORTS / "classification_metrics.csv", index=False)
    print("\nMetrics comparison:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Select best model by F1 score, which balances precision and recall.
    best_name = metrics_df.iloc[0]["Model"]
    best_model = fitted[best_name]
    best_pred = predictions[best_name]
    best_proba = probabilities[best_name]
    print(f"\nSelected best model by F1 Score: {best_name}")

    report = classification_report(
        y_test, best_pred, target_names=["Benign", "Malignant"], zero_division=0
    )
    (REPORTS / "classification_report.txt").write_text(
        f"Selected Model: {best_name}\n\n{report}\n", encoding="utf-8"
    )
    print("\nClassification report for selected model:")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, best_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(GRAPHS / "confusion_matrix_best_model.png", dpi=160)
    plt.close(fig)

    # ROC curve for selected model
    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_test, best_proba, ax=ax, name=best_name)
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    ax.set_title(f"ROC Curve - {best_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAPHS / "roc_curve_best_model.png", dpi=160)
    plt.close(fig)

    # Metric comparison chart
    plot_df = metrics_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]]
    ax = plot_df.plot(kind="bar", figsize=(12, 6), ylim=(0, 1.05))
    ax.set_title("Classification Metrics Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.legend(loc="lower right")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(GRAPHS / "model_metrics_comparison.png", dpi=160)
    plt.close()

    # Class distribution graph
    fig, ax = plt.subplots(figsize=(6, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Dataset Class Distribution")
    ax.set_xlabel("Class (0=Benign, 1=Malignant)")
    ax.set_ylabel("Number of samples")
    fig.tight_layout()
    fig.savefig(GRAPHS / "class_distribution.png", dpi=160)
    plt.close(fig)

    joblib.dump(best_model, ARTIFACTS / "best_classifier.joblib")
    (REPORTS / "summary.json").write_text(json.dumps({
        "dataset": "Breast Cancer Wisconsin (scikit-learn built-in)",
        "positive_class": "Malignant",
        "selected_model": best_name,
        "selection_metric": "F1 Score",
        "test_metrics": metrics_df.iloc[0].to_dict(),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "test_size": 0.20,
        "random_state": 42
    }, indent=2, default=float), encoding="utf-8")

    print("\nCompleted!")
    print("Saved reports and plots in:", REPORTS)
    print("Saved selected model in:", ARTIFACTS / "best_classifier.joblib")
    print("Graph files:")
    for p in sorted(GRAPHS.glob("*.png")):
        print(" -", p.name)

if __name__ == "__main__":
    main()
