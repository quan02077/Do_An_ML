import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


dataset = pd.read_csv('Bank Customer Churn Prediction.csv')

# Kham pha du lieu
# print(dataset.head())
# print(dataset.describe())
# print("Kich thuoc dataset: ", dataset.shape)
# print("Cac cot trong dataset: ", dataset.columns.tolist())
# print("Cac loai du lieu trong dataset: ", dataset.dtypes)
# print("Cac gia tri null trong dataset: ", dataset.isnull().sum())
# print("Cac gia tri trung lap trong dataset: ", dataset.duplicated().sum())

# # Tao bieu do de hien thi phan phoi cua bien muc tieu 'churn'
# print(dataset['churn'].value_counts())
# sns.countplot(x='churn', data=dataset, palette='coolwarm')
# plt.title('Phan phoi khach hang roi bo')
# plt.xlabel('Roi bo (1) / Khong roi bo (0)')
# plt.ylabel('So luong khach hang')
# plt.show()

labelEncoder = LabelEncoder()
categorical_cols = ['country', 'gender']
for col in categorical_cols:
    dataset[col] = labelEncoder.fit_transform(dataset[col])

X = dataset.drop(['customer_id', 'churn'], axis=1)
y = dataset['churn']


# print("Kich thuoc X:", X.shape)
# print("Kich thuoc y:", y.shape)
# print("Kieu du lieu cua X:", X.dtypes)
# print(X.head())
# print(y.head())   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# print("X_train:", X_train.shape)
# print("X_test:", X_test.shape)
# print("y_train:", y_train.shape)
# print("y_test:", y_test.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest
# model = RandomForestClassifier(random_state=0)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
# print("\nRecall:")
# print(recall_score(y_test, y_pred))
# print("\nPrecision:")
# print(precision_score(y_test, y_pred))
# print("\nF1 Score:")
# print(f1_score(y_test, y_pred))

# Logistic Regression
# model = LogisticRegression(random_state=0)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
# print("\nRecall:")
# print(recall_score(y_test, y_pred))
# print("\nPrecision:")
# print(precision_score(y_test, y_pred))
# print("\nF1 Score:")
# print(f1_score(y_test, y_pred))

#Naive Bayes
# model = GaussianNB()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
# print("\nRecall:")
# print(recall_score(y_test, y_pred))
# print("\nPrecision:")
# print(precision_score(y_test, y_pred))
# print("\nF1 Score:")
# print(f1_score(y_test, y_pred))

#Decision Tree
# model = DecisionTreeClassifier(random_state=0)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
# print("\nRecall:")
# print(recall_score(y_test, y_pred))
# print("\nPrecision:")
# print(precision_score(y_test, y_pred))
# print("\nF1 Score:")
# print(f1_score(y_test, y_pred))

# KNN
k = [3, 5, 7, 9, 11, 13, 15]
for i in k:
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nRecall:")
    print(recall_score(y_test, y_pred))
    print("\nPrecision:")
    print(precision_score(y_test, y_pred))
    print("\nF1 Score:")
    print(f1_score(y_test, y_pred))
    print("\n")