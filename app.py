import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.title("Machine Learning Classification Models")

col1, col2 = st.columns([3, 1])

with col1:
    st.write("Upload test dataset and evaluate different ML models.")

with col2:
    st.markdown(
        """
        **Name:** Danish Biyabani  
        **ID:** 2025AA05467
        """
    )

# --------------------------------------------------
# Load Artifacts
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("artifacts/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    with open("artifacts/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    models = {
        "Logistic Regression": pickle.load(open("artifacts/logistic_regression.pkl", "rb")),
        "Decision Tree": pickle.load(open("artifacts/decision_tree.pkl", "rb")),
        "KNN": pickle.load(open("artifacts/knn.pkl", "rb")),
        "Naive Bayes": pickle.load(open("artifacts/naive_bayes.pkl", "rb")),
        "Random Forest": pickle.load(open("artifacts/random_forest.pkl", "rb")),
        "XGBoost": pickle.load(open("artifacts/xgboost.pkl", "rb")),
    }

    return preprocessor, label_encoder, models


preprocessor, label_encoder, models = load_artifacts()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Model Selection")

selected_model_name = st.sidebar.selectbox(
    "Choose a model",
    list(models.keys())
)

model = models[selected_model_name]

st.subheader("Download Sample Test Dataset")

with open("test.csv", "rb") as f:
    st.download_button(
        label="Download test.csv",
        data=f,
        file_name="test.csv",
        mime="text/csv"
    )
# --------------------------------------------------
# File Upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Test CSV File (with target column)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=";")
    df.columns = df.columns.str.strip()

    TARGET_COL = "y"

    if TARGET_COL not in df.columns:
        st.error("Target column 'y' not found in uploaded CSV.")
        st.stop()

    X_test = df.drop(columns=[TARGET_COL])
    y_test = df[TARGET_COL]

    y_test_encoded = label_encoder.transform(y_test)

    # --------------------------------------------------
    # Preprocess
    # --------------------------------------------------
    X_test_processed = preprocessor.transform(X_test)

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------
    y_pred = model.predict(X_test_processed)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_processed)[:, 1]
        auc = roc_auc_score(y_test_encoded, y_prob)
    else:
        auc = np.nan

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    acc = accuracy_score(y_test_encoded, y_pred)
    prec = precision_score(y_test_encoded, y_pred)
    rec = recall_score(y_test_encoded, y_pred)
    f1 = f1_score(y_test_encoded, y_pred)
    mcc = matthews_corrcoef(y_test_encoded, y_pred)

    # --------------------------------------------------
    # Display Metrics
    # --------------------------------------------------
    st.subheader(f"Evaluation Metrics — {selected_model_name}")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{acc:.4f}")
    col1.metric("Precision", f"{prec:.4f}")

    col2.metric("Recall", f"{rec:.4f}")
    col2.metric("F1 Score", f"{f1:.4f}")

    col3.metric("AUC", f"{auc:.4f}")
    col3.metric("MCC", f"{mcc:.4f}")

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test_encoded, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

