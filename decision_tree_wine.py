import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# P1: Entropy

def entropy(counts):
    probabilities = counts / np.sum(counts)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))


counts = np.array([50, 30])

print("P1: Entropy")
print("Entropy:", entropy(counts))


# P2: Train Accuracy with Different Max Depths

iris = load_iris()
X = iris.data
y = iris.target

print("\nP2: Train Accuracy")

for depth in range(1, 9):
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    print("Depth:", depth, "Train Accuracy:", accuracy)


# Mini Project: Decision Tree on Wine Dataset

wine = load_wine()
X = wine.data
y = wine.target

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

depths = range(1, 9)
train_accuracies = []
val_accuracies = []

print("\nMini Project: Wine Dataset")

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)

    train_accuracy = accuracy_score(y_train, train_predictions)
    val_accuracy = accuracy_score(y_val, val_predictions)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print("Depth:", depth)
    print("Train Accuracy:", train_accuracy)
    print("Validation Accuracy:", val_accuracy)
    print("--------------------")


# Plot Train and Validation Accuracy

plt.figure(figsize=(8, 5))
plt.plot(depths, train_accuracies, marker="o", label="Train Accuracy")
plt.plot(depths, val_accuracies, marker="s", label="Validation Accuracy")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree - Wine Dataset")
plt.xticks(list(depths))
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("wine_accuracy.png")
plt.show()


# Best Depth

best_index = int(np.argmax(val_accuracies))
best_depth = list(depths)[best_index]

print("\nBest Depth:", best_depth)
print("Best Validation Accuracy:", val_accuracies[best_index])
