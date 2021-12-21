from datetime import datetime

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import os

cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train / 255., x_test / 255.


class Baseline(tf.keras.Model):
    def __init__(self):
        super(Baseline, self).__init__()
        # CBAPD套路
        self.c1 = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=1, padding='same')
        self.b1 = tf.keras.layers.BatchNormalization()
        self.a1 = tf.keras.layers.Activation('relu')
        self.p1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d1 = tf.keras.layers.Dropout(rate=0.2)

        self.flatten = tf.keras.layers.Flatten()
        self.f1 = tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())
        self.d2 = tf.keras.layers.Dropout(0.2)
        self.f2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        return y


model = Baseline()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['sparse_categorical_accuracy'])
check_point_path = './checkpoint/baseline.ckpt'
if os.path.exists(check_point_path):
    print('------------load model ---------')
    model.load_weights(filepath=check_point_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=check_point_path,
                                                 save_best_only=False,
                                                 save_weights_only=False)

log_dir_path = ".\\logs\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir_path, histogram_freq=1)

history = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), validation_freq=1,
                    batch_size=32, epochs=5, callbacks=[tb_callback])
model.summary()

# save_model_file = open('./savemodel/baselineweight.txt', 'w')
#
# for v in model.trainable_variables:
#     save_model_file.write(str(v.name) + '\n')
#     save_model_file.write(str(v.shape) + '\n')
#     save_model_file.write(str(v.numpy()) + '\n')
# save_model_file.close()

# sparse_categorical_accuracy 实际上是accuracy的别名
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# plot the result
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training And Validation Accuracy')
plt.show()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training And Validation Loss')
plt.show()
