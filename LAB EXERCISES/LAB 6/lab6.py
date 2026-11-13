# ==============================
# LAB 6 - EXPERIMENT 10
# EXPECTATION MAXIMIZATION
# ==============================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

X = np.concatenate([
    np.random.normal(5, 1, 100),
    np.random.normal(10, 1, 100)
]).reshape(-1, 1)

model = GaussianMixture(n_components=2, random_state=42)
model.fit(X)

labels = model.predict(X)

plt.scatter(X, np.zeros(len(X)), c=labels)
plt.title("Expectation Maximization Algorithm")
plt.xlabel("Data")
plt.show()

print("Cluster Labels:", labels)
print("Cluster Means:", model.means_)


# ==============================
# LAB 6 - EXPERIMENT 11
# CREDIT SCORE CLASSIFICATION
# ==============================

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(
    n_samples=200,
    n_features=4,
    n_informative=3,
    n_redundant=0,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))

print("Credit Score:", model.predict([[1, 0, 1, 0]]))


# ==============================
# LAB 6 - EXPERIMENT 12
# IRIS CLASSIFICATION USING KNN
# ==============================

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))

prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])

print("New Flower:", iris.target_names[prediction[0]])
