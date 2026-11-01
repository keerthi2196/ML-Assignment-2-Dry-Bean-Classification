from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
METADATA_PATH = ROOT / "model_metadata.json"

st.set_page_config(page_title="Dry Bean Classifier", page_icon="🫘", layout="wide")
st.title("Dry Bean Variety Classification")
st.write(
    "Upload a labelled CSV test file, choose one of the five trained classifiers, "
    "and review its evaluation metrics. The app loads saved models and does not retrain them."
)

if not METADATA_PATH.exists():
    st.error("Model files are not available. Run `python train.py` first and commit the generated model files.")
    st.stop()

metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
features = metadata["features"]
target = metadata["target"]

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

with st.sidebar:
    st.header("Evaluation settings")
    selected_model = st.selectbox("Select classifier", list(model_files.keys()))
    uploaded_file = st.file_uploader("Upload labelled test data (CSV)", type=["csv"])
    st.caption("CSV must contain the 16 feature columns and the target column `Class`.")

if uploaded_file is None:
    st.info("Upload `test_data.csv` from the project repository to display evaluation results.")
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the uploaded CSV: {exc}")
    st.stop()

required_columns = features + [target]
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    st.error("The uploaded file is missing required columns: " + ", ".join(missing_columns))
    st.stop()

if data.empty:
    st.error("The uploaded CSV contains no rows.")
    st.stop()

X_test = data[features].copy()
y_test = data[target].astype(str)

try:
    X_test = X_test.apply(pd.to_numeric, errors="raise")
except Exception:
    st.error("All 16 input feature columns must contain numeric values.")
    st.stop()

if X_test.isnull().any().any() or y_test.isnull().any():
    st.error("Missing values were found. Please upload a complete labelled test CSV.")
    st.stop()

model_path = MODEL_DIR / model_files[selected_model]
if not model_path.exists():
    st.error(f"Saved model not found: {model_path.name}. Run train.py and upload the generated models folder to GitHub.")
    st.stop()

model = joblib.load(model_path)
predictions = model.predict(X_test).astype(str)
probabilities = model.predict_proba(X_test)

try:
    auc = roc_auc_score(
        y_test, probabilities, labels=model.classes_,
        multi_class="ovr", average="weighted"
    )
except Exception as exc:
    st.error(
        "AUC could not be calculated. Ensure the uploaded test CSV contains examples "
        f"from all seven classes. Details: {exc}"
    )
    st.stop()

metrics = {
    "Accuracy": accuracy_score(y_test, predictions),
    "AUC (OvR)": auc,
    "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
    "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
    "F1 Score": f1_score(y_test, predictions, average="weighted", zero_division=0),
    "MCC": matthews_corrcoef(y_test, predictions)
}

st.subheader(f"Evaluation results — {selected_model}")
st.caption(f"Number of test records: {len(data):,}")

metric_cols = st.columns(3)
for col, (label, value) in zip(metric_cols * 2, metrics.items()):
    col.metric(label, f"{value:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    labels = list(model.classes_)
    cm = confusion_matrix(y_test, predictions, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Classification report")
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(4), use_container_width=True)

st.subheader("Prediction preview")
preview = X_test.head(10).copy()
preview["Actual Class"] = y_test.head(10).values
preview["Predicted Class"] = predictions[:10]
st.dataframe(preview, use_container_width=True)

st.download_button(
    "Download predictions as CSV",
    data=preview.to_csv(index=False).encode("utf-8"),
    file_name="dry_bean_predictions_preview.csv",
    mime="text/csv"
)
