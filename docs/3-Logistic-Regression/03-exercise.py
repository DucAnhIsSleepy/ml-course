import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay)

df = pd.read_csv(
    "hf://datasets/gymprathap/Chicago-Crime-Dataset/Crimes_-_2001_to_Present_20240713.csv",
    nrows=10000,
)

y = df["Arrest"]
x = df[["Primary Type", "Domestic", "Location Description", "District"]].fillna({
    "Primary Type": "Unknown",
    "Domestic": "Unknown", 
    "Location Description": "Unknown",
    "District": df["District"].median()
})

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["District"]),
        ("cat", OneHotEncoder(handle_unknown='ignore'), ["Primary Type", "Domestic", "Location Description"])
    ]
)

log_reg = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

log_reg.fit(x_train, y_train)

y_pred = log_reg.predict(x_test)
y_proba = log_reg.predict_proba(x_test)[:, 1]

results = pd.DataFrame({
    "y_test": y_test,
    "y_pred": y_pred,
    "y_proba": y_proba
})

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Arrest", "Arrest"]))