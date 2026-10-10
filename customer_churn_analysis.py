import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


dataset = pd.read_csv('Bank Customer Churn Prediction.csv')

# Khám phá dữ liệu
# print(dataset.head())
# print(dataset.describe())
# print("Kích thước dataset: ", dataset.shape)
# print("Các cột trong dataset: ", dataset.columns.tolist())
# print("Các loại dữ liệu trong dataset: ", dataset.dtypes)
# print("Các giá trị null trong dataset: ", dataset.isnull().sum())
# print("Các giá trị trùng lặp trong dataset: ", dataset.duplicated().sum())

# Tạo biểu đồ để hiển thị phân phối của biến mục tiêu 'churn'
# print(dataset['churn'].value_counts())
# sns.countplot(x='churn', data=dataset, palette='coolwarm')
# plt.title('Phân phối khách hàng rời bỏ')
# plt.xlabel('Rời bỏ (1) / Không rời bỏ (0)')
# plt.ylabel('Số lượng khách hàng')
# plt.show()

labelEncoder = LabelEncoder()
categorical_cols = ['country', 'gender']
for col in categorical_cols:
    dataset[col] = labelEncoder.fit_transform(dataset[col])

X = dataset.drop(['customer_id', 'churn'], axis=1)
y = dataset['churn']


# print("Kích thước X:", X.shape)
# print("Kích thước y:", y.shape)
# print("Kiểu dữ liệu của X:", X.dtypes)
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
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# Logistic Regression
# model = LogisticRegression(random_state=0)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

#Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred))