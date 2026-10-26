"""Practical Sheet 2: Linear Regression and Gradient Descent solutions."""
import numpy as np


def fit_line(x, y):
    """Return intercept theta0 and slope theta1 using least squares."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    theta1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    theta0 = y_mean - theta1 * x_mean
    return theta0, theta1


def exercises():
    # E1: Fit line to (0,1), (1,3), (2,4), (3,6).
    x = np.array([0, 1, 2, 3], dtype=float)
    y = np.array([1, 3, 4, 6], dtype=float)
    theta0, theta1 = fit_line(x, y)
    print("E1 theta0 (intercept):", round(theta0, 3))  # 1.100
    print("E1 theta1 (slope):", round(theta1, 3))      # 1.600

    # E2: J(theta)=(theta-5)^2, theta=0, learning rate=0.2.
    theta, eta = 0.0, 0.2
    gradient = 2 * (theta - 5)
    theta_new = theta - eta * gradient
    print("E2 theta after one step:", round(theta_new, 3))  # 2.000

    # P1: MSE of the fitted line on the sheet's data.
    x2 = np.array([1, 2, 3, 4], dtype=float)
    y2 = np.array([2, 2, 4, 5], dtype=float)
    t0, t1 = fit_line(x2, y2)
    y_hat = t0 + t1 * x2
    mse = np.mean((y2 - y_hat) ** 2)
    print("P1 training MSE:", round(mse, 3))  # 0.175

    # P2: Batch gradient descent for theta=[theta0, theta1].
    X = np.column_stack([np.ones(len(x2)), x2])
    theta = np.zeros(2)
    learning_rate = 0.01
    m = len(y2)
    for _ in range(20000):
        errors = X @ theta - y2
        gradient = (X.T @ errors) / m
        theta -= learning_rate * gradient
    print("P2 gradient-descent parameters [theta0, theta1]:", np.round(theta, 3))
    # Should approach [0.5, 1.1]


def california_housing_mini_project():
    """Compare Normal Equation and batch gradient descent, and plot two learning rates."""
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error

    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale input features using training data only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Add the intercept column.
    X_train_b = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
    X_test_b = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])

    # Normal Equation: theta = (X^T X)^(-1) X^T y.
    # np.linalg.pinv is numerically safer than explicitly calculating an inverse.
    theta_normal = np.linalg.pinv(X_train_b) @ y_train
    pred_normal = X_test_b @ theta_normal
    print("\nCalifornia Housing Mini-project")
    print("Normal Equation test MSE:", round(mean_squared_error(y_test, pred_normal), 4))

    # Batch gradient descent for two learning rates.
    m = len(y_train)
    learning_rates = [0.01, 0.1]
    histories = {}
    parameters = {}
    iterations = 300

    for eta in learning_rates:
        theta = np.zeros(X_train_b.shape[1])
        costs = []
        for _ in range(iterations):
            errors = X_train_b @ theta - y_train
            cost = np.mean(errors ** 2)
            costs.append(cost)
            gradient = (2 / m) * (X_train_b.T @ errors)
            theta -= eta * gradient
        histories[eta] = costs
        parameters[eta] = theta
        predictions = X_test_b @ theta
        print(f"Gradient Descent eta={eta} test MSE:",
              round(mean_squared_error(y_test, predictions), 4))

    plt.figure(figsize=(8, 5))
    for eta, costs in histories.items():
        plt.plot(range(1, iterations + 1), costs, label=f"learning rate={eta}")
    plt.xlabel("Iteration")
    plt.ylabel("Training MSE cost")
    plt.title("California Housing: Gradient Descent Cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig("california_housing_cost.png", dpi=150)
    plt.show()
    print("Cost plot saved as california_housing_cost.png")


if __name__ == "__main__":
    exercises()
    # The California Housing mini-project downloads a dataset and creates a plot.
    # Uncomment the next line to run it:
    # california_housing_mini_project()
