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

x_data = tf.cast(x_data, dtype=tf.float32)
y_data = tf.cast(y_data, dtype=tf.int32)


class Iris_Model(tf.keras.Model):
    def __init__(self, n):
        super(Iris_Model, self).__init__()
        self.d1 = tf.keras.layers.Dense(units=n, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, x):
        y = self.d1(x)
        return y

model = Iris_Model(3)

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
model.fit(x=x_data, y=y_data, epochs=500, batch_size=32, validation_split=0.2, validation_freq=20)

model.summary()
