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

# Decision Tree không cần StandardScaler; giữ dữ liệu gốc để ngưỡng dễ đọc.
model = DecisionTreeClassifier(max_depth=3, random_state=0, criterion="entropy")  
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Vẽ sơ đồ cây tương tự hình mẫu.
plt.figure(figsize=(24, 12))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Stay (0)", "Churn (1)"],
    filled=True,
    rounded=True,
    impurity=True,
    proportion=False,
    precision=3,
    fontsize=9,
)
plt.title("Decision Tree - Bank Customer Churn", fontsize=16)
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=200, bbox_inches="tight")
plt.show()