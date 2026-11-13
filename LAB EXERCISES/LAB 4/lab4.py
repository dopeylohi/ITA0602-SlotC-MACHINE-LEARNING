# ==============================
# LAB 4 - EXPERIMENT 6
# NAIVE BAYES ALGORITHM
# ==============================

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

X = [[2,60], [3,70], [4,80], [5,90],
     [1,40], [2,50], [6,95], [3,65]]

y = [0,0,1,1,0,0,1,0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=1, stratify=y
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0,1]))

print("Accuracy:", accuracy_score(y_test, y_pred))


# ==============================
# LAB 4 - EXPERIMENT 15
# IRIS FLOWER CLASSIFICATION
# ==============================

from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))


# ==============================
# LAB 4 - EXPERIMENT 19
# BANK LOAN PREDICTION
# ==============================

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

X = [
    [50000,700,0], [30000,600,1],
    [80000,750,0], [25000,550,1],
    [60000,720,0], [20000,500,1],
    [90000,800,0], [40000,650,1],
    [70000,740,0], [35000,620,1],
    [85000,780,0], [28000,570,1],
    [55000,710,0], [22000,520,1],
    [75000,760,0], [32000,590,1]
]

y = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))

result = model.predict([[55000, 710, 0]])

if result[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")


# ==============================
# LAB 4 - EXPERIMENT 20
# FUTURE SALES PREDICTION
# ==============================

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

months = np.array([[1], [2], [3], [4], [5], [6]])
sales = np.array([100, 200, 300, 400, 500, 600])

X_train, X_test, y_train, y_test = train_test_split(
    months, sales, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

future_month = np.array([[7]])
prediction = model.predict(future_month)

print("Predicted Sales:", prediction[0])
