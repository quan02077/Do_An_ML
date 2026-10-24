import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from DecisionTree_srcatch import DecisionTree

# Đọc dữ liệu (đặt file CSV cùng thư mục với file Python này)
data = pd.read_csv("Bank Customer Churn Prediction.csv")

label_encoder = LabelEncoder()
categorical_cols = ["country", "gender"]
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

X = data.drop(["customer_id", "churn"], axis=1)
y = data["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

dt_scratch = DecisionTree(max_depth=3)
dt_scratch.fit(X_train, y_train)
y_pred_scratch = dt_scratch.predict(X_test)
print("Accuracy (Decision Tree from scratch):", accuracy_score(y_test, y_pred_scratch))

clf_plot = DecisionTreeClassifier(max_depth=3, criterion="entropy", random_state=0)
clf_plot.fit(X_train, y_train)
print("Accuracy (sklearn):", accuracy_score(y_test, clf_plot.predict(X_test)))


plt.figure(figsize=(22, 11))
plot_tree(
    clf_plot,
    feature_names=X.columns.tolist(),
    class_names=["Stay (0)", "Churn (1)"],
    filled=True,
    rounded=True,
    impurity=True,
    precision=3,
    fontsize=9,
)
plt.title("Sơ đồ Cây Quyết Định (Decision Tree) - Bank Customer Churn", fontsize=16)
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=300, bbox_inches="tight")
print("-> Da luu so do cay vao file: decision_tree.png")
plt.show()
