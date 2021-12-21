# 导入模块
import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

data = datasets.load_iris()

x_data = data.data
y_data = data.target

np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

x_train = x_data[:-30]
x_test = x_data[-30:]
y_train = y_data[:-30]
y_test = y_data[-30:]

x_train = tf.cast(x_train, dtype=tf.float32)
x_test = tf.cast(x_test, dtype=tf.float32)

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# create model
w1 = tf.Variable(tf.random.truncated_normal(shape=[4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal(shape=[3], stddev=0.1, seed=1))

lr = 0.1
epochs = 500
train_loss_results = []
test_acc = []
loss_all = 0

m_w, m_b = 0, 0
beta = 0.9

now_time = time.time()

for epoch in range(epochs):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_train, depth=3)
            # 当前这一点对训练样的损失
            loss = tf.reduce_mean(tf.square(y_ - y))
            loss_all += loss.numpy()
        grads = tape.gradient(loss, [w1, b1])

        m_w = beta * m_w + (1 - beta) * grads[0]
        m_b = beta * m_b + (1 - beta) * grads[1]
        w1.assign_sub(lr * m_w)
        b1.assign_sub(lr * m_b)
    print("Epoch: {}, loss: {}".format(epoch, loss_all / 4))

    train_loss_results.append(loss_all / 4)
    loss_all = 0

    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)
        pred = tf.cast(pred, dtype=y_test.dtype)
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += correct
        total_number += x_test.shape[0]  # batch
    acc = total_correct / total_number
    test_acc.append(acc)
    print('Test_acc', acc)
    print("----------------------")

total_time = time.time() - now_time
print('total_time', total_time)

plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results, label='$Loss$')
plt.legend()
plt.show()

plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(test_acc, label='Accuracy')
plt.legend()
plt.show()
