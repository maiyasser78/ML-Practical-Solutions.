"""Practical Sheet 0: Python refresher solutions."""
import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def exercises():
    # P1: Keep even numbers from the given list.
    nums = [3, 1, 4, 1, 5, 9, 2, 6]
    evens = [x for x in nums if x % 2 == 0]
    print("P1 even numbers:", evens)  # [4, 2, 6]

    # P2: Create an array from 1 to 5, multiply by 2, and find its mean.
    a = np.arange(1, 6)
    print("P2 a * 2:", a * 2)        # [ 2  4  6  8 10]
    print("P2 mean:", a.mean())      # 3.0

    # P3: Repeat the Iris workflow using DecisionTreeClassifier.
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    model = make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=0))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("P3 Decision Tree test accuracy:", round(accuracy_score(y_test, predictions), 3))


def mini_project():
    # Mini-project: load a built-in dataset, split, scale, train a classifier, report accuracy.
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000)
    )
    model.fit(X_train, y_train)
    print("Mini-project dataset: Wine")
    print("Mini-project test accuracy:", round(model.score(X_test, y_test), 3))


if __name__ == "__main__":
    exercises()
    print()
    mini_project()
