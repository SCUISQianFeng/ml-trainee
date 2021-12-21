# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
import os

# pre
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# build model
model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    layers.Conv2D(filters=32, activation='relu', kernel_size=(3, 3), kernel_regularizer=keras.regularizers.l2(l=0.01),
                  padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(filters=64, activation='relu',kernel_size=(3, 3), kernel_regularizer=keras.regularizers.l2(l=0.01),
                  padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(filters=128, activation='relu',kernel_size=(3, 3), kernel_regularizer=keras.regularizers.l2(l=0.01),
                  padding='same'),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(units=64, activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.01)),
    layers.Dropout(0.5),
    layers.Dense(units=10)
])

def my_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), kernel_regularizer=keras.regularizers.l2(l=0.01), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_regularizer=keras.regularizers.l2(l=0.01), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_regularizer=keras.regularizers.l2(l=0.01), padding='same')(
        x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.01))(x)
    x = layers.Dropout(rate=0.5)(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# model = my_model()

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(lr=3e-4),
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=64, epochs=100, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)
