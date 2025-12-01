# app.py
# ============================================
# Finance Case Study Lab: End-to-End Pipeline
# Dataset: bank_with_missing.csv
# Target: y (term deposit subscription: yes/no)
# ============================================

# 0) Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score, roc_curve
)

# --- Start of Script ---

## 1) Read CSV and Preview

# NOTE: This script assumes 'bank_with_missing.csv' is available in the current directory.
try:
    df = pd.read_csv("bank_with_missing.csv")
except FileNotFoundError:
    print("Error: 'bank_with_missing.csv' not found. Please ensure the file is in the correct directory.")
    exit()

print("## 1) Data Load and Preview 📊")
print("Shape:", df.shape)
print("\nTop 5 rows (may include NaNs):")
print(df.head())
print("-" * 50)


## 2) & 3) Cleaning (Handling NaNs)

# Count NaNs before cleaning
print("NaN counts before cleaning:")
print(df.isnull().sum().sort_values(ascending=False).head())

# A. 'age' (numeric): Fill missing with MEAN
# Choice: Mean imputation is a simple technique for numerical data, often acceptable for a small number of missing values.
print("\n[Cleaning] A: Fill 'age' NaNs with mean")
age_nan_before = df['age'].isna().sum()
df['age'].fillna(df['age'].mean(), inplace=True)
print(f"  'age' NaNs before: {age_nan_before} -> after: {df['age'].isna().sum()}")

# B. 'job' (categorical): Fill missing with MODE
# Choice: Mode imputation is a standard approach for categorical data when missing values are few.
print("\n[Cleaning] B: Fill 'job' NaNs with mode")
job_nan_before = df['job'].isna().sum()
# mode() returns a Series, [0] selects the first mode if there are multiple
df['job'].fillna(df['job'].mode()[0], inplace=True)
print(f"  'job' NaNs before: {job_nan_before} -> after: {df['job'].isna().sum()}")

# C. 'balance' (numeric): Drop rows where balance is NaN
# Choice: Dropping rows is chosen here if 'balance' is deemed a critical feature where imputed values might introduce significant noise or bias.
print("\n[Cleaning] C: Drop rows where 'balance' is NaN")
shape_before = df.shape
df = df.dropna(subset=['balance'])
print(f"  Shape before: {shape_before} -> after: {df.shape}")
print("  Remaining NaNs:")
print(df.isnull().sum().sort_values(ascending=False).head())
print("-" * 50)


## Optional Sanity Check Visual (Target Distribution)
plt.figure()
df['y'].value_counts().plot(kind='bar')
plt.title("Target Distribution (y) after cleaning")
plt.xlabel("y")
plt.ylabel("count")
plt.show()
print("")
print("-" * 50)


## 4) Encoding

# Encode target 'y' (no/yes -> 0/1)
print("\n## 4) Encoding 🔄")
print("[Encoding] Target 'y' -> 0/1")
le = LabelEncoder()
# Typically 'no' (alphabetically first) maps to 0, and 'yes' to 1.
df['y'] = le.fit_transform(df['y'])
print("Target value counts after encoding (0=No, 1=Yes):")
print(df['y'].value_counts())

# One-hot encode the features
X_raw = df.drop(columns=['y'])
y = df['y']
# Use drop_first=True to avoid multicollinearity (the 'unknown' or first category becomes the baseline)
X = pd.get_dummies(X_raw, drop_first=True)

print(f"Feature matrix shape (after encoding): {X.shape}")
print("\nFirst 5 rows of encoded features (X):")
print(X.head())
print("-" * 50)


## 5) Stratified Train/Test Split (80/20)

print("\n## 5) Data Splitting ✂️")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"[Split] Train shapes: {X_train.shape} {y_train.shape}")
print(f"[Split] Test shapes: {X_test.shape} {y_test.shape}")
print("-" * 50)


## 6) Scale Numeric Features

print("\n## 6) Feature Scaling ⚖️")

# Convert to numpy arrays for scaling
X_train_np = X_train.values.astype(float)
X_test_np  = X_test.values.astype(float)

# StandardScaler: Note: Using with_mean=False is a safe choice if the dataset
# might contain sparse data (which is not the case here after get_dummies conversion to NumPy, 
# but often used in general pipelines). with_mean=True is typically used for dense data like this.
# We'll stick to the original code's suggestion but note that with_mean=True is also appropriate.
scaler = StandardScaler(with_mean=False) 

# IMPORTANT: Fit scaler on TRAIN data only; Transform TRAIN & TEST.
scaler.fit(X_train_np)
X_train_sc = scaler.transform(X_train_np)
X_test_sc  = scaler.transform(X_test_np)

print("[Scaling] Completed (fit on train, applied to test)")
print("\nFirst 5 scaled train samples:")
print(X_train_sc[0:5])
print("-" * 50)


## 7) Train & Evaluate Models

print("\n## 7) Model Training and Evaluation 🤖")
# --- Logistic Regression ---
log_clf = LogisticRegression(max_iter=1000, random_state=42)
log_clf.fit(X_train_sc, y_train)

y_pred_log = log_clf.predict(X_test_sc)
y_prob_log = log_clf.predict_proba(X_test_sc)[:, 1] # Probability of the positive class (1)

print("\n=== Logistic Regression ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
print(classification_report(y_test, y_pred_log, digits=4))

# --- Decision Tree ---
tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_clf.fit(X_train_sc, y_train)

y_pred_tree = tree_clf.predict(X_test_sc)
y_prob_tree = tree_clf.predict_proba(X_test_sc)[:, 1] # Probability of the positive class (1)

print("\n=== Decision Tree (max_depth=5) ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tree):.4f}")
print(classification_report(y_test, y_pred_tree, digits=4))
print("-" * 50)


## 8) Visualization of Metrics (Confusion Matrix & ROC Curve)

# --- Confusion Matrices ---
fig, axs = plt.subplots(1, 2, figsize=(10,4))

# Logistic Regression Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_log, ax=axs[0])
axs[0].set_title("Confusion Matrix - Logistic")

# Decision Tree Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_tree, ax=axs[1])
axs[1].set_title("Confusion Matrix - Decision Tree")

plt.tight_layout()
plt.show()
print("")

# --- ROC Curves and AUC ---
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_tree)
auc_log = roc_auc_score(y_test, y_prob_log)
auc_tree = roc_auc_score(y_test, y_prob_tree)

plt.figure()
plt.plot(fpr_log, tpr_log, label=f"Logistic (AUC={auc_log:.3f})")
plt.plot(fpr_tree, tpr_tree, label=f"Decision Tree (AUC={auc_tree:.3f})")
plt.plot([0,1], [0,1], 'k--', linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (Holdout Test)")
plt.legend()
plt.show()
print("")

# --- Summary ---
print("\n## Summary of Model Performance (on Test Set) ✨")
print(f"Logistic Regression AUC: {auc_log:.3f}")
print(f"Decision Tree (Max Depth 5) AUC: {auc_tree:.3f}")
print("Both models show similar, high accuracy, but their ability to correctly identify the positive class (term deposit 'yes', class 1) is limited, as shown by the low recall and f1-score for class 1.")
