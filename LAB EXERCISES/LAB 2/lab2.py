# ==============================
# LAB 2 - EXPERIMENT 3
# DECISION TREE ID3
# ==============================

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = {
    'Outlook': ['Sunny', 'Sunny', 'Rainy', 'Sunny', 'Rainy', 'Sunny'],
    'Temperature': ['Warm', 'Warm', 'Cold', 'Cold', 'Warm', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'Normal', 'Normal', 'High'],
    'Wind': ['Strong', 'Weak', 'Strong', 'Weak', 'Weak', 'Strong'],
    'Play': ['Yes', 'Yes', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

X = pd.get_dummies(df.drop('Play', axis=1))
y = df['Play']

model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

new_data = pd.DataFrame([{
    'Outlook': 'Sunny',
    'Temperature': 'Warm',
    'Humidity': 'Normal',
    'Wind': 'Strong'
}])

new_data = pd.get_dummies(new_data)
new_data = new_data.reindex(columns=X.columns, fill_value=0)

print("Prediction:", model.predict(new_data)[0])
print("Accuracy:", model.score(X, y))


# ==============================
# LAB 2 - EXPERIMENT 4
# ANN USING BACKPROPAGATION
# ==============================

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MLPClassifier(
    hidden_layer_sizes=(10,),
    max_iter=2000,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
