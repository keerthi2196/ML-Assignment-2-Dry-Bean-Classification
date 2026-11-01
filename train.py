from pathlib import Path
import json
import time
import joblib
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, classification_report, confusion_matrix
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
TARGET = "Class"

def load_dataset():
    """Download the official UCI Dry Bean dataset and save a local CSV copy."""
    dataset = fetch_ucirepo(id=602)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    y.columns = [TARGET]
    data = pd.concat([X, y], axis=1)
    data.to_csv(ROOT / "dry_bean_dataset.csv", index=False)
    return X, y[TARGET], data

def build_models():
    """Pipelines ensure scaling is learned from training data only."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Gaussian Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB())
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
        ])
    }

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(
            y_test, probabilities, labels=model.classes_,
            multi_class="ovr", average="weighted"
        ),
        "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "F1": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions)
    }

def make_readme(metrics_df, feature_names, class_names, n_samples):
    winner = metrics_df.sort_values("Accuracy", ascending=False).iloc[0]
    rows = []
    for _, r in metrics_df.iterrows():
        rows.append(
            f"| {r['Model']} | {r['Accuracy']:.4f} | {r['AUC']:.4f} | "
            f"{r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1']:.4f} | {r['MCC']:.4f} |"
        )
    observations = []
    for _, r in metrics_df.iterrows():
        observations.append(
            f"| {r['Model']} | Accuracy {r['Accuracy']:.4f}, weighted F1 {r['F1']:.4f}, "
            f"AUC {r['AUC']:.4f}, and MCC {r['MCC']:.4f}. |"
        )
    readme = f"""# Dry Bean Variety Classification

## a. Problem statement
The objective is to classify a dry bean sample into one of seven varieties using 16 numerical geometric and shape-related measurements extracted from bean images. Five supervised classification algorithms are trained and compared using the same stratified train/test split.

## b. Dataset description
- **Dataset:** Dry Bean
- **Source:** UCI Machine Learning Repository (Dataset ID: 602)
- **Samples:** {n_samples:,}
- **Input features:** {len(feature_names)}
- **Target:** `Class`
- **Classes:** {", ".join(map(str, class_names))}
- **Missing values:** The UCI dataset reports no missing values.
- **Dataset citation:** Koklu, M. and Ozkan, I.A. (2020), *Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques*, Computers and Electronics in Agriculture. UCI DOI: https://doi.org/10.24432/C50S4B

## c. GitHub Repository Link
Add your repository URL here: **[PASTE GITHUB REPOSITORY URL]**

## d. Models used and evaluation metrics
An 80:20 stratified train/test split was used with `random_state=42`. Precision, recall, and F1 are weighted averages across the seven classes. AUC is calculated using the one-vs-rest (OvR) multiclass method with weighted averaging.

| ML Model Name | Accuracy | AUC (OvR) | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

### Observations on model performance
| ML Model Name | Observation about model performance |
|---|---|
{chr(10).join(observations)}
| **Overall winner (highest accuracy)** | **{winner['Model']}** with accuracy {winner['Accuracy']:.4f}. Consider AUC, F1 and MCC as supporting measures when interpreting the result. |

## Implementation details
- Logistic Regression: StandardScaler + LogisticRegression.
- Decision Tree: DecisionTreeClassifier.
- KNN: StandardScaler + KNeighborsClassifier (k=5).
- Naive Bayes: StandardScaler + GaussianNB.
- SVM: StandardScaler + RBF-kernel SVC with probability estimates enabled.
- Scaling is fitted inside each pipeline using training data only, preventing test-data leakage.

## Streamlit application
Live app URL: **[PASTE STREAMLIT COMMUNITY CLOUD URL]**

The app accepts a labelled test CSV, lets the evaluator select one of the five saved models, and displays accuracy, AUC, precision, recall, F1, MCC, a confusion matrix, and a classification report. The app does not retrain models on uploaded test data.

## How to run locally
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Upload `test_data.csv` from this repository to evaluate the saved models in the app.
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")

def main():
    print("Loading Dry Bean dataset from UCI...")
    X, y, full_data = load_dataset()
    print(f"Dataset shape: {full_data.shape}")
    print(f"Number of input features: {X.shape[1]}")
    print(f"Classes: {sorted(y.unique().tolist())}")
    print("Class distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # Include the target column: the Streamlit app evaluates labelled test rows.
    test_data = X_test.copy()
    test_data[TARGET] = y_test
    test_data.to_csv(ROOT / "test_data.csv", index=False)

    results = []
    for name, model in build_models().items():
        print(f"\nTraining {name}...")
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        scores = evaluate_model(model, X_test, y_test)
        scores["Model"] = name
        scores["Training Time (seconds)"] = elapsed
        results.append(scores)

        safe_name = (
            name.lower().replace(" ", "_").replace("-", "_")
            .replace("(", "").replace(")", "")
        )
        joblib.dump(model, MODEL_DIR / f"{safe_name}.joblib")

        print(f"Training time: {elapsed:.2f} seconds")
        print(classification_report(y_test, model.predict(X_test), zero_division=0))

    metrics_df = pd.DataFrame(results)[
        ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC", "Training Time (seconds)"]
    ].sort_values("Accuracy", ascending=False)
    metrics_df.to_csv(RESULT_DIR / "model_comparison.csv", index=False)

    metadata = {
        "target": TARGET,
        "features": X.columns.tolist(),
        "classes": sorted(y.unique().tolist()),
        "random_state": RANDOM_STATE,
        "test_size": 0.20
    }
    (ROOT / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    make_readme(metrics_df, X.columns.tolist(), sorted(y.unique().tolist()), len(full_data))

    print("\n========== MODEL COMPARISON ==========")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved models in ./models/")
    print("Saved labelled test data as ./test_data.csv")
    print("Saved comparison table as ./results/model_comparison.csv")
    print("Generated README.md with your actual evaluation results.")

if __name__ == "__main__":
    main()
