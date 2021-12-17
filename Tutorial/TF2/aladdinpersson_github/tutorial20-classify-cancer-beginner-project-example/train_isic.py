# -*- coding:utf-8 -*-

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import numpy as np
import math
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_examples = 20225
test_examples = 2551
validation_examples = 2555
img_height = img_width = 224
batch_size = 32

model = keras.Sequential([
    hub.KerasLayer(handle='https://hub.tensorflow.google.cn/google/imagenet/nasnet_mobile/feature_vector/4',
                   trainable=True),
    layers.Dense(1, activation='sigmoid')
])

train_datagen = ImageDataGenerator(rotation_range=15,
                                   rescale=1.0 / 255,
                                   zoom_range=(0.95, 0.95),
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   data_format='channels_last',
                                   dtype=tf.float32)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255, dtype=tf.float32)
test_datagen = ImageDataGenerator(rescale=1.0 / 255, dtype=tf.float32)

train_gen = train_datagen.flow_from_directory(directory='data/train/',
                                              target_size=(img_height, img_width),
                                              color_mode='rgb',
                                              class_mode='binary',
                                              batch_size=batch_size,
                                              shuffle=True,
                                              seed=2021)

validation_gen = validation_datagen.flow_from_directory(directory='data/validation/',
                                                        target_size=(img_height, img_width),
                                                        color_mode='rgb',
                                                        class_mode='binary',
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        seed=2021)

test_gen = test_datagen.flow_from_directory(directory='data/test/',
                                            target_size=(img_height, img_width),
                                            color_mode='rgb',
                                            class_mode='binary',
                                            batch_size=batch_size,
                                            shuffle=True,
                                            seed=2021)

METRICS = [
    keras.metrics.BinaryCrossentropy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc')
]

model.compile(
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
    optimizer=keras.optimizers.Adam(lr=3e-4),
    metrics=METRICS
)
tb_calback = keras.callbacks.TensorBoard(log_dir="tb_callback", histogram_freq=1)
model.fit(train_gen,
          epochs=1,
          verbose=2,
          steps_per_epoch=train_examples // batch_size,
          validation_data=validation_gen,
          validation_split=validation_examples // batch_size,
          callbacks=[tb_calback])


def plot_auc(labels, data):
    predictions = model.predict(data)
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    plt.plot(100 * fpr, 100 * tpr)
    plt.xlabel('False positive [%]')
    plt.ylabel('True positive [%]')
    plt.show()


test_labels = np.arary([])
num_batches = 0

for _, y in test_gen:
    test_labels = np.append(test_labels, y)
    num_batches += 1
    if num_batches == math.ceil(test_examples / batch_size):
        break

plot_auc(test_labels, test_gen)
model.evaluate(validation_gen, verbose=2)
model.evaluate(test_gen, verbose=2)

print(model.summary())
import sys

sys.exit()
