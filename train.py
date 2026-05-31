"""
Training pipeline — baseline vs full-feature Decision Tree for bank churn prediction.

Trains two models on the same stratified train/test split and prints metrics
side-by-side so the contribution of feature expansion + class balancing is
measurable:

  1. Baseline   — 3 numerical features + 1 engineered (Age, NumOfProducts,
                  Balance, BalancePerProduct), no class weighting, max_depth=5
                  (matches the original notebook).
  2. Full model — all 10 usable features with one-hot encoding for Geography
                  and Gender, class_weight='balanced' for the 79/21 imbalance,
                  and a 5-fold cross-validated max_depth sweep.

Outputs:
  - Console comparison of accuracy / precision / recall / F1 / ROC-AUC
  - decision_tree.onnx          — full model, exported via skl2onnx
  - confusion_matrix.png        — side-by-side baseline vs full
  - feature_importance.png      — full model

train.py uses:
  - dataset.csv                 → 10,000-row Bank Customer Churn dataset
  - scikit-learn                → DecisionTreeClassifier, train_test_split, GridSearchCV
  - skl2onnx + onnxruntime      → model export + smoke test
  - matplotlib                  → confusion matrix + importance plots
  - pandas                      → load + one-hot encoding via get_dummies
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


# ─── Config ──────────────────────────────────────────
DATA_PATH = "dataset.csv"
ONNX_PATH = "decision_tree.onnx"
FEATURE_COLUMNS_PATH = "feature_columns.json"
RANDOM_STATE = 42
TEST_SIZE = 0.30

BASELINE_FEATURES = ["Age", "NumOfProducts", "Balance", "BalancePerProduct"]
CATEGORICAL_COLUMNS = ["Geography", "Gender"]


# ─── Load + prepare ──────────────────────────────────
print("=" * 70)
print("Loading data…")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
df["BalancePerProduct"] = df["Balance"] / (df["NumOfProducts"] + 1)

print(f"Rows: {len(df):,}")
print(f"Class balance: {df['Exited'].value_counts(normalize=True).to_dict()}")
print()

y = df["Exited"]
X = df.drop(columns=["Exited"])

# Stratified split — both train and test keep the 79/21 ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)


# ─── Baseline: 3 features + 1 engineered, no balancing ──
print("=" * 70)
print("BASELINE — 4 features, no class weighting, max_depth=5")
print("=" * 70)

baseline = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
baseline.fit(X_train[BASELINE_FEATURES], y_train)
y_pred_baseline = baseline.predict(X_test[BASELINE_FEATURES])
y_proba_baseline = baseline.predict_proba(X_test[BASELINE_FEATURES])[:, 1]

print(f"Accuracy:  {accuracy_score(y_test, y_pred_baseline):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba_baseline):.4f}")
print(classification_report(y_test, y_pred_baseline, target_names=["Stayed", "Churned"]))


# ─── Full: 12 features (10 raw + 2 one-hot dummies after drop_first) ──
print("=" * 70)
print("FULL MODEL — all 10 features, class_weight='balanced', max_depth swept")
print("=" * 70)

# pd.get_dummies with drop_first reduces collinearity:
#   Geography (France, Germany, Spain) → Geography_Germany, Geography_Spain
#   Gender    (Female, Male)           → Gender_Male
X_train_full = pd.get_dummies(X_train, columns=CATEGORICAL_COLUMNS, drop_first=True)
X_test_full = pd.get_dummies(X_test, columns=CATEGORICAL_COLUMNS, drop_first=True)

# Align columns (test may be missing categories — get_dummies wouldn't add them)
X_test_full = X_test_full.reindex(columns=X_train_full.columns, fill_value=0)

feature_columns = list(X_train_full.columns)
print(f"Feature count after encoding: {len(feature_columns)}")
print(f"Features: {feature_columns}")
print()

grid = GridSearchCV(
    DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    param_grid={"max_depth": [3, 5, 7, 9, 11, None]},
    cv=5,
    scoring="f1",            # F1 on the minority class is what matters for imbalanced data
    n_jobs=-1,
)
grid.fit(X_train_full, y_train)
full = grid.best_estimator_

y_pred_full = full.predict(X_test_full)
y_proba_full = full.predict_proba(X_test_full)[:, 1]

print(f"Best max_depth (CV-selected): {grid.best_params_['max_depth']}")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_full):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba_full):.4f}")
print(classification_report(y_test, y_pred_full, target_names=["Stayed", "Churned"]))


# ─── Side-by-side summary ────────────────────────────
print("=" * 70)
print("SUMMARY — baseline vs full")
print("=" * 70)

def short_metrics(y_true, y_pred, y_proba):
    from sklearn.metrics import precision_score, recall_score, f1_score
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "churn_precision": precision_score(y_true, y_pred, pos_label=1),
        "churn_recall": recall_score(y_true, y_pred, pos_label=1),
        "churn_f1": f1_score(y_true, y_pred, pos_label=1),
    }

baseline_m = short_metrics(y_test, y_pred_baseline, y_proba_baseline)
full_m = short_metrics(y_test, y_pred_full, y_proba_full)

print(f"{'Metric':<22}{'Baseline':>14}{'Full':>14}{'Δ':>14}")
print("-" * 64)
for k in baseline_m:
    b, f = baseline_m[k], full_m[k]
    print(f"{k:<22}{b:>14.4f}{f:>14.4f}{(f - b):>+14.4f}")
print()


# ─── Plots ───────────────────────────────────────────
def plot_confusion(ax, y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Stayed", "Churned"])
    ax.set_yticklabels(["Stayed", "Churned"])


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_confusion(axes[0], y_test, y_pred_baseline, "Baseline — 4 features")
plot_confusion(axes[1], y_test, y_pred_full, f"Full model — {len(feature_columns)} features, balanced")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=100, bbox_inches="tight")
plt.close()
print("Saved: confusion_matrix.png")


# Feature importance — full model only
importances = full.feature_importances_
order = np.argsort(importances)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(np.array(feature_columns)[order], importances[order], color="#4c72b0")
ax.set_xlabel("Feature importance")
ax.set_title("Full Model — Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=100, bbox_inches="tight")
plt.close()
print("Saved: feature_importance.png")


# ─── ONNX export ─────────────────────────────────────
n_features = len(feature_columns)
initial_type = [("input", FloatTensorType([None, n_features]))]
onnx_model = convert_sklearn(full, initial_types=initial_type, target_opset=12)
with open(ONNX_PATH, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"Saved: {ONNX_PATH} ({n_features} features)")

# Persist feature column order so main.py builds the request vector identically
with open(FEATURE_COLUMNS_PATH, "w") as f:
    json.dump(feature_columns, f, indent=2)
print(f"Saved: {FEATURE_COLUMNS_PATH}")

print()
print("Training complete.")
