# -*- coding:utf-8 -*-

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.
x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.
#
# model = keras.Sequential()
#
# model.add(keras.Input(shape=(28 * 28)))
# model.add(layers.Dense(units=512, activation='relu'))
# model.add(layers.Dense(units=256, activation='relu'))
# model.add(layers.Dense(10))

inputs = keras.Input(shape=(784))
x = layers.Dense(units=512, activation='relu', name='first_layer')(inputs)
x = layers.Dense(units=256, activation='relu', name='second_layer')(x)
outputs = layers.Dense(units=10, activation='softmax', )(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=128, verbose=2)  # verbose 简单打印
model.evaluate(x_test, y_test, batch_size=128, verbose=2)
