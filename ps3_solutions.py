"""Practical Sheet 3: Regularization and Logistic Regression solutions."""
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error


def exercises():
    # E1: w=(0,-2,5), alpha=0.2
    w = np.array([0.0, -2.0, 5.0])
    alpha = 0.2
    l1_penalty = alpha * np.sum(np.abs(w))
    l2_penalty = alpha * np.sum(w ** 2)
    print("E1 L1 penalty:", round(l1_penalty, 3))  # 1.4
    print("E1 L2 penalty:", round(l2_penalty, 3))  # 5.8

    # E2: z=-0.4; p=sigmoid(z); predict class 1 if p >= 0.5.
    z = -0.4
    probability = 1 / (1 + np.exp(-z))
    predicted_class = int(probability >= 0.5)
    print("E2 probability:", round(float(probability), 3))  # 0.401
    print("E2 predicted class:", predicted_class)           # 0


def softmax(z):
    """Convert scores to probabilities that sum to 1."""
    z = np.asarray(z, dtype=float)
    shifted = z - np.max(z)  # numerical stability
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def python_exercises():
    # P1: Softmax for [2.0, 1.0, 0.1].
    print("P1 softmax:", np.round(softmax([2.0, 1.0, 0.1]), 3))
    # Expected: [0.659, 0.242, 0.099]

    # P2: Fit Ridge(alpha=1.0) and Lasso on the same generated data.
    rng = np.random.RandomState(0)
    X = rng.randn(50, 5)
    y = X @ np.array([3.0, 0.0, 0.0, 2.0, 0.0]) + 0.1 * rng.randn(50)
    lasso = Lasso(alpha=0.1).fit(X, y)
    ridge = Ridge(alpha=1.0).fit(X, y)
    print("P2 Lasso coefficients:", np.round(lasso.coef_, 2))
    print("P2 Ridge coefficients:", np.round(ridge.coef_, 2))
    print("Comparison: Ridge usually shrinks coefficients toward zero but generally does not make them exactly zero; Lasso can set some coefficients exactly to zero.")


def diabetes_mini_project():
    """Compare Linear Regression, Ridge, and Lasso on the Diabetes dataset."""
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge alpha=0.1": Ridge(alpha=0.1),
        "Ridge alpha=1.0": Ridge(alpha=1.0),
        "Ridge alpha=10.0": Ridge(alpha=10.0),
        "Lasso alpha=0.01": Lasso(alpha=0.01, max_iter=20000),
        "Lasso alpha=0.1": Lasso(alpha=0.1, max_iter=20000),
        "Lasso alpha=1.0": Lasso(alpha=1.0, max_iter=20000),
    }

    print("\nDiabetes Mini-project (features scaled inside each pipeline)")
    for name, estimator in models.items():
        model = make_pipeline(StandardScaler(), estimator)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = model.score(X_test, y_test)
        fitted_estimator = model.named_steps[estimator.__class__.__name__.lower()]
        zero_features = []
        if isinstance(estimator, Lasso):
            feature_names = load_diabetes().feature_names
            zero_features = [
                feature_names[i]
                for i, coef in enumerate(fitted_estimator.coef_)
                if abs(coef) < 1e-8
            ]
        print(f"{name}: test MSE={mse:.3f}, R^2={r2:.3f}")
        if isinstance(estimator, Lasso):
            print("  Features set to zero:", zero_features if zero_features else "None")


if __name__ == "__main__":
    exercises()
    python_exercises()
    diabetes_mini_project()
