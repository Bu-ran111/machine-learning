import os
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# CIFAR-10测试集的文件路径
test_batches_dir = r'C:\Users\ASUS\Desktop\机器学习\大作业\cifar-10-python\cifar-10-batches-py'
test_batch_files = [os.path.join(test_batches_dir, f) for f in os.listdir(test_batches_dir) if 'test_batch' in f]

# 加载CIFAR-10测试集（加载所有10000个图像）
images = []
labels = []
for batch_file in test_batch_files:
    with open(batch_file, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')  # CIFAR-10数据是以pickle格式存储的
        images.append(batch['data'])
        labels.append(batch['labels'])
images = np.concatenate(images)
labels = np.concatenate(labels)

# 将图像数据转换为浮点数并归一化到 [0, 1] 范围
images = images.astype(np.float32) / 255.0

# 使用PCA降维
n_components = 100  # 降到100维（
pca = PCA(n_components=n_components)
reduced_images = pca.fit_transform(images)

# 应用K-means进行分类
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(reduced_images)
cluster_labels = kmeans.labels_

label_counts = np.zeros((n_clusters, 10), dtype=int)  # 10是CIFAR-10的类别数
for i in range(n_clusters):
    cluster_labels_in_cluster = labels[cluster_labels == i]
    label_counts[i, :] = np.bincount(cluster_labels_in_cluster, minlength=10)
most_common_labels = np.argmax(label_counts, axis=1)
predicted_labels = most_common_labels[cluster_labels]
accuracy_variant = accuracy_score(labels, predicted_labels) * 100  # 乘以100得到百分比

# 计算每个类别的召回率（准确率的变种）和错误率
recall_per_class = np.zeros(10)
error_rate_per_class = np.zeros(10)
for true_label in range(10):
    true_label_indices = (labels == true_label)
    predicted_as_true_label = (predicted_labels[true_label_indices] == true_label)
    recall_per_class[true_label] = np.mean(predicted_as_true_label) * 100  # 召回率
    error_rate_per_class[true_label] = 100 - recall_per_class[true_label]  # 错误率

# 输出结果
print(f'准确率变种（非真实准确率）: {accuracy_variant:.2f}%')
print("每个类别的召回率（准确率的变种）和错误率：")
for i in range(10):
    print(f'类别 {i} 召回率: {recall_per_class[i]:.2f}%, 错误率: {error_rate_per_class[i]:.2f}%')

# 可视化聚类结果
num_images_to_visualize = 10000  # 要可视化的图像数量（在降维后的空间中）
pca_for_visualization = PCA(n_components=2)  # 再次降到2维以便可视化
pca_result_for_visualization = pca_for_visualization.fit_transform(reduced_images[:num_images_to_visualize])
plt.scatter(pca_result_for_visualization[:, 0], pca_result_for_visualization[:, 1], c=cluster_labels[:num_images_to_visualize], s=50, cmap='viridis')
plt.colorbar(ticks=range(n_clusters))
plt.title('K-means ')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

# 计算混淆矩阵
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(labels, predicted_labels)

# 可视化混淆矩阵
import seaborn as sns
import pandas as pd

# 将混淆矩阵转换为DataFrame以便更好地可视化
cm_df = pd.DataFrame(cm, index=range(10), columns=range(10))

# 使用seaborn绘制混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for K-means Clustering on CIFAR-10')
plt.show()