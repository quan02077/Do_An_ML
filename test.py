import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from DecisionTree_srcatch import DecisionTree

def accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total

data = pd.read_csv("Bank Customer Churn Prediction.csv")

labelEncoder = LabelEncoder()
categorical_cols = ['country', 'gender']
for col in categorical_cols:
    data[col] = labelEncoder.fit_transform(data[col])

X = data.drop(['customer_id', 'churn'], axis=1)
y = data['churn']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = DecisionTree(max_depth=5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy(y_test, y_pred)

print("Accuracy:", acc)