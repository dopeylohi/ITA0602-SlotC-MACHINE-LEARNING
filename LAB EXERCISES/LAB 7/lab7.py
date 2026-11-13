# ==============================
# LAB 7 - EXPERIMENT 13
# CAR PRICE PREDICTION
# ==============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = {
    "Year": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    "Mileage": [50000, 45000, 40000, 35000, 30000, 25000, 20000, 15000],
    "Engine": [1000, 1200, 1200, 1500, 1500, 1800, 1800, 2000],
    "Price": [300000, 350000, 400000, 500000, 550000, 650000, 750000, 850000]
}

df = pd.DataFrame(data)

X = df[["Year", "Mileage", "Engine"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predicted Prices:", y_pred)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

new_car = [[2023, 10000, 2000]]

print("Car Price:", model.predict(new_car)[0])


# ==============================
# LAB 7 - EXPERIMENT 14
# HOUSE PRICE PREDICTION
# ==============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = {
    "Area": [1000, 1200, 1500, 1800, 2000, 2200, 2500, 3000],
    "Bedrooms": [2, 2, 3, 3, 3, 4, 4, 5],
    "Age": [10, 8, 5, 7, 4, 3, 2, 1],
    "Price": [2000000, 2500000, 3000000, 3500000,
              4000000, 4500000, 5000000, 6000000]
}

df = pd.DataFrame(data)

X = df[["Area", "Bedrooms", "Age"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predicted Prices:", y_pred)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

new_house = [[2000, 3, 5]]

print("House Price:", model.predict(new_house)[0])
