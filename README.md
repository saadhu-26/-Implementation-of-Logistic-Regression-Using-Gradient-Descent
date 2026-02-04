# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:

To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load the given dataset and preprocess the data.

2.Initialize weights, bias, learning rate, and number of iterations.

3.Apply Gradient Descent to update weights using logistic loss function.

4.Predict the output and display the result. 

## Program:
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: 
RegisterNumber:  
*/
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv(r"C:\Users\acer\Downloads\Placement_Data.csv")
print("Dataset Loaded Successfully\n")

le = LabelEncoder()

categorical_columns = [
    "gender",
    "ssc_b",
    "hsc_b",
    "hsc_s",
    "degree_t",
    "workex",
    "specialisation",
    "status"
]

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

X = df.drop(["status", "salary", "sl_no"], axis = 1).values
y = df["status"].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

X_train = (X_train - X_train.mean(axis = 0)) / X_train.std(axis = 0)
X_test  = (X_test  - X_test.mean(axis = 0))  / X_test.std(axis = 0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent(X, y, learning_rate, iterations):

    m, n = X.shape
    W = np.zeros((n, 1))
    b = 0

    for i in range(iterations):

        Z = np.dot(X, W) + b
        A = sigmoid(Z)

        dW = (1 / m) * np.dot(X.T, (A - y))
        db = (1 / m) * np.sum(A - y)

        W = W - learning_rate * dW
        b = b - learning_rate * db

    return W, b

learning_rate = 0.01
iterations = 1000

W, b = gradient_descent(X_train, y_train, learning_rate, iterations)

def predict(X, W, b):

    Z = np.dot(X, W) + b
    A = sigmoid(Z)
    return (A >= 0.5).astype(int)

y_pred = predict(X_test, W, b)

accuracy = np.mean(y_pred == y_test)

print("Model Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

cr = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(cr)
```

## Output:

<img width="421" height="288" alt="Screenshot 2026-02-04 184237" src="https://github.com/user-attachments/assets/a9b37c8f-e471-404f-8106-0222d7f83b12" />

## Result:

Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
