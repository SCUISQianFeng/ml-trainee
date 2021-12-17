# -*- coding:utf-8 -*-
import tensorflow as tf
import os
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# HYPERPARAMETERS
BATCH_SIZE = 64
WEIGHT_DECAY = 0.001
LEARNING_RATE = 0.001
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
train_images = os.getcwd() + "/train_images/" + train_df.iloc[:, 0].values
test_images = os.getcwd() + "/test_images/" + test_df.iloc[:, 0].values
train_labels = train_df.iloc[:, 1:].values
test_labels = test_df.iloc[:, 1:].values


def read_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_path, channels=1, dtype=tf.float32)
    image.set_shape(64, 64, 1)
    label[0].set_shape([])
    label[1].set_shape([])

    labels = {'first_nm': label[0], 'second_num': label[1]}
    return image, labels


AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = {
    train_dataset.shuffle(buffer_size=len(train_labels))
        .map(read_image())
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
}

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = {
    test_dataset.map(read_image())
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
}

inputs = keras.Input(shape=(64, 64, 1))
x = layers.Conv2D(filters=32,
                  kernel_size=(3, 3),
                  padding='same',
                  kernel_regularizer=keras.regularizers.l2(l=0.01))(inputs)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.Conv2D(filters=64,
                  kernel_size=(3, 3),
                  padding='same',
                  kernel_regularizer=keras.regularizers.l2(l=0.01))(x)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(filters=128,
                  kernel_size=(3, 3),
                  padding='same',
                  kernel_regularizer=keras.regularizers.l2(l=0.01))(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout()(x)
x = layers.Dense(64, activation='relu')(x)
output1 = layers.Dense(10, activation='relu', name='first_num')(x)
output2 = layers.Dense(10, activation='relu', name='second_num')(x)

model = keras.Model(inputs=inputs, output=[output1, output2])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(LEARNING_RATE),
    metrics=['accuracy']
)
model.fit(train_images, train_labels, epochs=5, batch_size=64, verbose=2)
model.evaluate(test_images, test_labels, batch_size=64, verbose=2)








