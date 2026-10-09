import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('Bank Customer Churn Prediction.csv')

print(dataset.head())
print(dataset.isnull().sum())
print(dataset.describe())

print(dataset['churn'].value_counts())
sns.countplot(x='churn', data=dataset, palette='coolwarm')
plt.title('Phân phối khách hàng rời bỏ')
plt.xlabel('Rời bỏ (1) / Không rời bỏ (0)')
plt.ylabel('Số lượng khách hàng')
plt.show()