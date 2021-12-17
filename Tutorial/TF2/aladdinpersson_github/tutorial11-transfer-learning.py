# -*- coding:utf-8 -*-
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow_hub as hub

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.
x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.

x = tf.random.normal(shape=(3, 299, 299, 3))
y = tf.constant([0, 1, 2])
new_model = keras.applications.InceptionV3(include_top=True)

base_inputs = new_model.layers[0].input
base_output = new_model.layers[-2].output
classifier = layers.Dense(3)(base_output)
new_model = keras.Model(inputs=base_inputs, outputs=classifier)

new_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

print(new_model.summary())
new_model.fit(x, y, epochs=15, verbose=2)


url = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4"
base_model = hub.KerasLayer(url=url, input_shape=(299, 299, 3))
model = keras.Sequential([
    base_model,
    layers.Dense(128, activation='relu'),
    layers.Dense(164, activation='relu'),
    layers.Dense(10),
])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy']
)


model.fit(x, y, batch_size=32, epochs=15, verbose=2)

