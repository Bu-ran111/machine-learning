import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# 加载CIFAR-10数据集
def load_cifar_10(data_dir):
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(data_dir, f'data_batch_{b}')
        with open(f, 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
            xs.append(d[b'data'])
            ys.append(d[b'labels'])

    x = np.concatenate(xs)
    y = np.concatenate(ys)

    x = x.reshape((x.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1).astype('float32') / 255.0
    x = x.reshape((x.shape[0], -1))  # Flatten the images to 1D vectors

    x_test = []
    y_test = []
    f = os.path.join(data_dir, 'test_batch')
    with open(f, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
        x_test.append(d[b'data'])
        y_test.append(d[b'labels'])

    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)

    x_test = x_test.reshape((x_test.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1).astype('float32') / 255.0
    x_test = x_test.reshape((x_test.shape[0], -1))  # Flatten the images to 1D vectors

    return shuffle(x, y, random_state=42), shuffle(x_test, y_test, random_state=42)

# 数据目录
data_dir = r'C:\Users\ASUS\Desktop\机器学习\大作业\cifar-10-python\cifar-10-batches-py'

# 加载数据
(x_train, y_train), (x_test, y_test) = load_cifar_10(data_dir)

# 分割训练集为训练集和验证集
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# 定义要测试的树的数量范围
n_estimators_range = [10, 50, 100, 200, 500]
val_accuracies = []
val_errors = []

# 对每个数量的树进行训练和验证
for n_estimators in n_estimators_range:
    rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, verbose=1)
    rf.fit(x_train, y_train)
    y_val_pred = rf.predict(x_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_error = 1 - val_accuracy
    val_accuracies.append(val_accuracy)
    val_errors.append(val_error)

# 绘制验证准确率与树的数量之间的关系
plt.figure()
plt.plot(n_estimators_range, val_accuracies, marker='o', label='Validation Accuracy')
plt.plot(n_estimators_range, val_errors, marker='x', label='Validation Error')
plt.xlabel('Number of Trees')
plt.ylabel('Metric Value')
plt.title('Validation Metrics vs. Number of Trees in Random Forest')
plt.legend()
plt.grid(True)
plt.show()

# 选择最佳数量的树，并在测试集上进行最终评估
best_n_estimators = n_estimators_range[np.argmax(val_accuracies)]
rf_best = RandomForestClassifier(n_estimators=best_n_estimators, n_jobs=-1, verbose=1)
rf_best.fit(np.concatenate([x_train, x_val]), np.concatenate([y_train, y_val]))  # 使用全部训练数据重新训练
y_test_pred = rf_best.predict(x_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_error = 1 - test_accuracy

# 计算召回率
recall_macro = recall_score(y_test, y_test_pred, average='macro')
recall_micro = recall_score(y_test, y_test_pred, average='micro')

# 输出结果
print(f'Test Accuracy with Best Number of Trees ({best_n_estimators}): {test_accuracy:.4f}')
print(f'Test Error with Best Number of Trees ({best_n_estimators}): {test_error:.4f}')
print(f'Macro-average Recall: {recall_macro:.4f}')
print(f'Micro-average Recall: {recall_micro:.4f}')

# 绘制测试准确率（可选）
plt.figure()
plt.bar(['Test Accuracy'], [test_accuracy], color='blue')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.xlabel('Dataset')
plt.title('CIFAR-10 Classification with Random Forest (Best Number of Trees)')
plt.show()

# 输出混淆矩阵
conf_matrix = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=range(10))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()