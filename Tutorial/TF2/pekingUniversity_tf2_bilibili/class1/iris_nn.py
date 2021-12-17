# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# %matplotlib.inline

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ['CPP_TF_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# load dataset
data = load_iris()
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3)
# convert numpy to tf
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# 批处理
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 截断正态分布，保证生成的数据在mean的两倍std内，超过范围的重新生成
w1 = tf.Variable(tf.random.truncated_normal([4, 3], mean=0, stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], mean=0, stddev=0.1, seed=1))
lr = 0.1
train_loss_result = []
test_acc = []
epochs = 500
loss_all = 0

# 训练部分
for epoch in range(epochs):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_train, depth=3)
            loss = tf.reduce_mean(tf.square(y_ - y))
            loss_all += loss
        grads = tape.gradient(loss, [w1, b1])
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])



    print('epoch: {}, loss: {}'.format(epoch, loss_all / 4))
    train_loss_result.append(loss_all / 4)
    loss_all = 0  # reset loss_all

    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        y_pred = tf.matmul(x_test, w1) + b1
        y_pred = tf.nn.softmax(y_pred)
        pred = tf.argmax(y_pred, axis=1)
        pred = tf.cast(pred, dtype=y_test.dtype)
        correct = tf.cast(tf.equal(y_test, pred), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += x_test.shape[0]  # m * n 中的m行
    acc = total_correct / total_number
    test_acc.append(acc)
    print('Test_acc', acc)
    print("-------------------")

# 绘制 loss 曲线
plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_result, label="$Loss$")
plt.legend()
plt.show()

# 绘制 Accuracy 曲线
plt.title('Acc Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Acc')  # y轴变量名称
plt.plot(test_acc, label="$Accuracy$")  # 逐点画出test_acc值并连线，连线图标是Accuracy
plt.legend()
plt.show()

np.random.rand()