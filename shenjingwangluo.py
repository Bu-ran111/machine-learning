import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from sklearn.metrics import classification_report, accuracy_score

# CIFAR-10 数据集的本地路径
cifar10_dir = r'C:\Users\ASUS\Desktop\机器学习\大作业\cifar-10-python\cifar-10-batches-py'


# 定义读取 CIFAR-10 二进制文件的函数
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# 读取训练和测试批次
def load_cifar10(data_dir):
    xs = []
    ys = []
    for b in range(1, 6):  # CIFAR-10 训练集有5个批次
        f = os.path.join(data_dir, f'data_batch_{b}')
        d = unpickle(f)
        xs.append(d[b'data'])
        ys.append(d[b'labels'])

    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)

    # CIFAR-10 测试集有1个批次
    f = os.path.join(data_dir, 'test_batch')
    d = unpickle(f)
    x_test = d[b'data']
    y_test = d[b'labels']

    # CIFAR-10 数据集图像是3072维的（32x32x3），需要重塑为(32, 32, 3)
    x_train = x_train.reshape((x_train.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1).astype("float32")
    x_test = x_test.reshape((x_test.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1).astype("float32")

    # 归一化像素值到0-1之间
    x_train /= 255.0
    x_test /= 255.0

    return (x_train, y_train), (x_test, y_test)


# 加载 CIFAR-10 数据集
(train_images, train_labels), (test_images, test_labels) = load_cifar10(cifar10_dir)

# 定义类名
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 构建卷积神经网络模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])

# 打印模型摘要
model.summary()

# 编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 使用 tf.data.Dataset 创建数据集
BATCH_SIZE = 64

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(
    buffer_size=len(train_images)).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(BATCH_SIZE).prefetch(
    buffer_size=tf.data.AUTOTUNE)

# 训练模型
history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# 绘制训练 & 验证的准确率值
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# 评估模型在测试集上的表现
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test accuracy: {test_acc}')

# 获取预测结果
predictions = model.predict(test_images).argmax(axis=1)

# 打印分类报告，包括每个类别的准确率
print(classification_report(test_labels, predictions, target_names=class_names))

# 绘制前10个测试样本的预测结果
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                        100 * np.max(predictions_array),
                                        class_names[true_label]),
               color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# 绘制测试样本及其预测结果
num_rows = 5
num_cols = 2
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, model.predict(test_images[i:i+1]), test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, model.predict(test_images[i:i+1]), test_labels)
plt.tight_layout()
plt.show()