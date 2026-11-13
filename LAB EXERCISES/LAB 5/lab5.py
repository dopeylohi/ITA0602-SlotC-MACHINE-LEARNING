# ==============================
# LAB 5 - EXPERIMENT 5
# K-NEAREST NEIGHBOURS
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

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))


# ==============================
# LAB 5 - EXPERIMENT 7
# LOGISTIC REGRESSION
# ==============================

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))


# ==============================
# LAB 5 - EXPERIMENT 8
# LINEAR REGRESSION
# ==============================

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Prediction for 11:", model.predict([[11]]))


# ==============================
# LAB 5 - EXPERIMENT 9
# LINEAR AND POLYNOMIAL REGRESSION
# ==============================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([2, 5, 10, 17, 26, 37, 50, 65])

linear = LinearRegression()
linear.fit(X, y)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

y_linear = linear.predict(X)
y_poly = model.predict(X_poly)

plt.scatter(X, y, color="red", label="Actual")
plt.plot(X, y_linear, label="Linear Regression")
plt.plot(X, y_poly, label="Polynomial Regression")

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

print("Linear Predictions:", y_linear)
print("Polynomial Predictions:", y_poly)
