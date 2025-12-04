
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
import xgboost as xgb


def evaluate_model(y_test, y_pred, y_probs, model_name):
    """Utility to compute and print common metrics."""
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n=== {model_name} ===")
    print("Confusion Matrix:\n", cm)
    print(f"ROC AUC: {auc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

    return {
        "model": model_name,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


def plot_importance(importances, cols, title):
    plt.figure(figsize=(8, 4))
    plt.bar(cols, importances)
    plt.xticks(rotation=90)
    plt.ylabel("Feature importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def proba_model(model, X, y, method="predict_proba"):
    probs = cross_val_predict(
        model,
        X,
        y,
        cv=5,
        method=method
    )[:,1]
    return probs
