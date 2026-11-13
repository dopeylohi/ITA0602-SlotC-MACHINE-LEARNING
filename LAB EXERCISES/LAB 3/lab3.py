# ==============================
# LAB 3 - EXPERIMENT 16
# CLASSIFICATION ALGORITHMS COMPARISON
# ==============================

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(name, "Accuracy:", accuracy_score(y_test, y_pred))


# ==============================
# LAB 3 - EXPERIMENT 17
# MOBILE PRICE PREDICTION
# ==============================

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = np.array([
    [2, 16, 2000, 5],
    [3, 32, 2500, 8],
    [4, 64, 3000, 12],
    [4, 128, 3500, 16],
    [6, 128, 4000, 32],
    [6, 256, 4500, 48],
    [8, 256, 5000, 64],
    [8, 512, 6000, 108],
    [2, 32, 2200, 8],
    [4, 64, 3200, 16],
    [6, 128, 4200, 32],
    [8, 512, 5500, 64]
])

y = np.array([0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))

new_mobile = [[6, 128, 4000, 48]]

print("Mobile Price Range:", model.predict(new_mobile)[0])


# ==============================
# LAB 3 - EXPERIMENT 18
# PERCEPTRON IRIS CLASSIFICATION
# ==============================

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Perceptron(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))

new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("Flower Type:", iris.target_names[prediction[0]])
