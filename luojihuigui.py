import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, recall_score, classification_report
import matplotlib.pyplot as plt
from glob import glob
import pickle

# CIFAR-10 数据集文件夹路径
data_dir = r"C:\Users\ASUS\Desktop\机器学习\大作业\cifar-10-python\cifar-10-batches-py"

# 辅助函数：读取 CIFAR-10 二进制文件
def unpickle(file_path):
    with open(file_path, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict

# 加载 CIFAR-10 数据集
def load_cifar_10(data_dir):
    X_train, y_train = [], []
    for i in range(1, 6):
        file_path = os.path.join(data_dir, f'data_batch_{i}')
        data_dict = unpickle(file_path)
        X_train.append(data_dict[b'data'])
        y_train.append(data_dict[b'labels'])

    X_test = unpickle(os.path.join(data_dir, 'test_batch')).get(b'data')
    y_test = unpickle(os.path.join(data_dir, 'test_batch')).get(b'labels')

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    y_train = np.array(y_train).astype(int)  # 直接转换为整数
    y_test = np.array(y_test).astype(int)  # 直接转换为整数

    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    return (X_train, y_train), (X_test, y_test)

# 使用 PCA 降维
def apply_pca(X, n_components):
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X)
    return pca.transform(X)

# 逻辑回归分类
def train_logistic_regression(X_train, y_train, C=1.0, max_iter=1000):
    clf = LogisticRegression(C=C, max_iter=max_iter, solver='liblinear', multi_class='ovr')
    clf.fit(X_train, y_train)
    return clf

# 主函数
def main():
    # 加载数据集
    (X_train, y_train), (X_test, y_test) = load_cifar_10(data_dir)

    # 使用全部训练数据
    X_train_small = X_train[:10000]
    y_train_small = y_train[:10000]

    # 应用 PCA 降维
    n_components = 100
    X_train_pca = apply_pca(X_train_small, n_components)
    X_test_pca = apply_pca(X_test, n_components)  # 对全部测试数据应用PCA

    # 训练逻辑回归模型
    clf = train_logistic_regression(X_train_pca, y_train_small, max_iter=2000)  # 增加迭代次数以提高收敛性

    # 对测试集进行分类
    y_pred = clf.predict(X_test_pca)

    # 输出准确率
    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1 - accuracy
    print(f'准确率: {accuracy:.2f}, 错误率: {error_rate:.2f}')

    # 输出召回率和其他分类指标
    report = classification_report(y_test, y_pred, output_dict=True)
    macro_recall = report['weighted avg']['recall']
    print(f'宏平均召回率: {macro_recall:.2f}')

    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))

    plt.figure(figsize=(15, 15))  # 增大图形大小以适应混淆矩阵
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for 10000 Test Samples')
    plt.xticks(rotation=90)  # 旋转x轴标签以便更好地显示
    plt.yticks(rotation=0)  # 保持y轴标签水平
    plt.tight_layout()  # 调整布局以避免重叠
    plt.show()

if __name__ == "__main__":
    main()