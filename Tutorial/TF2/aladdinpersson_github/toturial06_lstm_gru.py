# -*- coding:utf-8 -*-

import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

model = keras.Sequential([
    keras.Input(shape=(None, 28)),
    layers.SimpleRNN(units=512, return_sequences=True, activation='relu'),
    layers.SimpleRNN(units=512, activation='relu'),
    layers.Dense(10)
])


model = keras.Sequential()
model.add(keras.Input(shape=(None, 28)))
model.add(layers.GRU(units=256, activation='tanh', return_sequences=True))
model.add(layers.GRU(units=256))
model.add(layers.Dense(10))

model = keras.Sequential()
model.add(keras.Input(shape=(None, 28)))
model.add(layers.LSTM(units=256, activation='tanh', return_sequences=True))
model.add(layers.LSTM(units=256))
model.add(layers.Dense(10))

model = keras.Sequential()
model.add(keras.Input(shape=(32, 32)))
model.add(layers.Bidirectional(
    layers.LSTM(units=256, activation='relu', return_sequences=True))
)
model.add(layers.Bidirectional(layers.LSTM(units=256, name='lstm_layer2')))
model.add(layers.Dense(10))

print(model.summary())
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(lr=0.01),
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=2)
model.evaluate(x_test, y_test, batch_size=128, verbose=2)
