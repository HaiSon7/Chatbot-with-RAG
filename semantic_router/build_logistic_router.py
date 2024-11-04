import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib  # Thư viện để lưu mô hình

# Đọc dữ liệu từ tệp CSV
data = pd.read_csv('embeddings_sample.csv')

# Kiểm tra kiểu dữ liệu của cột 'embedding'
print(data['embedding'].head())

# Chuyển cột 'embedding' từ chuỗi thành vector
# Chuyển đổi với json.loads nếu dữ liệu là chuỗi JSON
import json
X = np.array([json.loads(x) for x in data['embedding']])  # Hoặc giữ eval() nếu chắc chắn về dữ liệu
y = data['label'].values  # Chuyển thành numpy array

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra với tỷ lệ 80-20 và phân tầng theo nhãn
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Kiểm tra kích thước sau khi chia
print("Kích thước X_train:", X_train.shape)
print("Kích thước y_train:", y_train.shape)
print("Tỷ lệ nhãn trong tập train:\n", np.unique(y_train, return_counts=True))
print("Kích thước X_test:", X_test.shape)
print("Kích thước y_test:", y_test.shape)
print("Tỷ lệ nhãn trong tập test:\n", np.unique(y_test, return_counts=True))

# Tạo mô hình hồi quy logistic
model = LogisticRegression()

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Lưu mô hình
joblib.dump(model, 'logistic_model.pkl')

'''
Accuracy: 0.9943502824858758
Confusion Matrix:
 [[100   1]
 [  0  76]]
Classification Report:
               precision    recall  f1-score   support

           0       1.00      0.99      1.00       101
           1       0.99      1.00      0.99        76

    accuracy                           0.99       177
   macro avg       0.99      1.00      0.99       177
weighted avg       0.99      0.99      0.99       177
'''