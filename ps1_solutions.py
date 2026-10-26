"""Practical Sheet 1: Classification solutions."""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)


def exercises():
    # E1: TP=30, FN=20, FP=10, TN=40
    tp, fn, fp, tn = 30, 20, 10, 40
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print("E1 Accuracy:", round(accuracy, 3))    # 0.700
    print("E1 Precision:", round(precision, 3))  # 0.750
    print("E1 Recall:", round(recall, 3))        # 0.600
    print("E1 F1:", round(f1, 3))                # 0.667

    # E2: One-vs-All needs K classifiers; One-vs-One needs K(K-1)/2.
    k = 5
    print("E2 One-vs-All classifiers:", k)
    print("E2 One-vs-One classifiers:", k * (k - 1) // 2)

    # P1: Confusion matrix for the labels from the sheet's Python Example 1.
    y_true = [1, 1, 1, 0, 0, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
    print("P1 confusion matrix (rows=true [0,1], columns=predicted [0,1]):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    # [[4 1]
    #  [1 4]]


def mini_project():
    # Mini-project: Logistic Regression on Breast Cancer, 80/20 split, scaled.
    # In sklearn, target 0=malignant and 1=benign.
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000)
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    default_predictions = (probabilities >= 0.5).astype(int)
    print("\nMini-project: Breast Cancer Logistic Regression")
    print("Confusion matrix at threshold 0.5 (labels [0,1]):")
    print(confusion_matrix(y_test, default_predictions, labels=[0, 1]))
    print("F1 at threshold 0.5:", round(f1_score(y_test, default_predictions), 3))
    print("Recall for class 1 at threshold 0.5:", round(recall_score(y_test, default_predictions), 3))

    # Lower threshold for class 1 generally increases recall for class 1.
    threshold = 0.35
    lower_predictions = (probabilities >= threshold).astype(int)
    print(f"\nConfusion matrix at lower threshold {threshold} (labels [0,1]):")
    print(confusion_matrix(y_test, lower_predictions, labels=[0, 1]))
    print("F1 at lower threshold:", round(f1_score(y_test, lower_predictions), 3))
    print("Recall for class 1 at lower threshold:", round(recall_score(y_test, lower_predictions), 3))
    print(
        "Note: sklearn labels are 0=malignant, 1=benign. "
        "This threshold experiment raises recall for class 1 (benign). "
        "If the goal is to raise recall for malignant cases, define malignant as the positive class "
        "or use the probability of class 0 and adjust its threshold."
    )


if __name__ == "__main__":
    exercises()
    mini_project()
