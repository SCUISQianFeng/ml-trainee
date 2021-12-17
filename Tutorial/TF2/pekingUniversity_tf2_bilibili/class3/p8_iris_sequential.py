import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import datasets

data = datasets.load_iris()

x_data = data.data
y_data = data.target

np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
# x_train = x_data[:-30]
# x_test = x_data[-30:]
# y_train = y_data[:-30]
# y_test = y_data[-30:]
#
# x_train = tf.cast(x_train, dtype=tf.float32)
# x_test = tf.cast(x_test, dtype=tf.float32)

x_data = tf.cast(x_data, dtype=tf.float32)
y_data = tf.cast(y_data, dtype=tf.int32)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x=x_data, y=y_data, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)
model.summary()
