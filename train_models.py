import os
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

from xgboost import XGBClassifier

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------
DATA_PATH = "bank-marketing.csv"
df = pd.read_csv(DATA_PATH, sep=";")
df.columns = df.columns.str.strip()

# --------------------------------------------------
# 2. Separate features & target
# --------------------------------------------------
TARGET_COL = "y"

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# --------------------------------------------------
# 3. Identify categorical & numerical columns
# --------------------------------------------------
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# --------------------------------------------------
# 4. Preprocessing Pipeline
# --------------------------------------------------
numeric_transformer = StandardScaler()
categorical_transformer = categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# --------------------------------------------------
# 5. Train-test split (STRATIFIED)
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# --------------------------------------------------
# 6. Models dictionary
# --------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced"
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42, class_weight="balanced"
    ),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}

# --------------------------------------------------
# 7. Train models & evaluate
# --------------------------------------------------
results = []

os.makedirs("artifacts", exist_ok=True)

# Fit preprocessor ONLY on training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Save preprocessor & label encoder
with open("artifacts/preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

with open("artifacts/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

for name, model in models.items():
    print(f"Training {name}...")

    model.fit(X_train_processed, y_train)

    y_pred = model.predict(X_test_processed)

    # For AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_processed)[:, 1]
    else:
        y_proba = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

    results.append([
        name, acc, auc, prec, rec, f1, mcc
    ])

    # Save model
    file_name = name.lower().replace(" ", "_") + ".pkl"
    with open(f"artifacts/{file_name}", "wb") as f:
        pickle.dump(model, f)

# --------------------------------------------------
# 8. Save metrics table
# --------------------------------------------------
metrics_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
)

metrics_df.to_csv("artifacts/metrics.csv", index=False)

print("\nTraining completed successfully!")
print(metrics_df)


test_df = X_test.copy()
test_df[TARGET_COL] = label_encoder.inverse_transform(y_test)

os.makedirs("data", exist_ok=True)
test_df.to_csv("test.csv", index=False, sep=";")

print("test.csv saved successfully")
