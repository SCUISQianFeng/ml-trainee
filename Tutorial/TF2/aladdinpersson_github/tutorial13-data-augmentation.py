# -*- coding:utf-8 -*-
import os

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(ds_train, ds_test), ds_info = tfds.load(
    name='cifar10',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def normalize_img(image, label):
    """Normalizes images"""
    return tf.cast(image, tf.float32) / 255.0, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32


def augment(image, label):
    new_width = new_height = 32
    image = tf.image.resize(images=image, size=(new_height, new_width))

    if tf.random.uniform((), minval=0, maxval=1) < 0.1:
        # rgb_to_grayscale: 3 channels(rgb) to 1 channel(gray)
        # tf.tile(x, [1,1,3]） 将x的最后一个维度复制3次
        image = tf.tile(tf.image.rgb_to_grayscale(images=image), [1, 1, 3])
    image = tf.image.random_brightness(image=image, max_delta=0.1)  # 随机亮度
    image = tf.image.random_contrast(image=image, lower=0.1, upper=0.2)  # 随机对比度
    image = tf.image.random_flip_left_right(image)  # 随机镜像
    return image, label


ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.map(augment)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.prefetch(AUTOTUNE)

# 新版无此用法
"""
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.Resizing(height=32, width=32,),
        layers.experimental.preprocessing.RandomFlip(mode="horizontal"),
        layers.experimental.preprocessing.RandomContrast(factor=0.1,),
    ]
)
"""

model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    # data_augmentation,
    # layers.Conv2D(4, 3, padding="same", activation="relu"),
    # layers.Conv2D(8, 3, padding="same", activation="relu"),
    # layers.MaxPooling2D(),
    # layers.Conv2D(16, 3, padding="same", activation="relu"),
    # layers.Flatten(),
    # layers.Dense(64, activation="relu"),
    # layers.Dense(10),
    layers.Conv2D(filters=32, kernel_size=(3, 3),
                  kernel_regularizer=keras.regularizers.l2(l=0.01),
                  padding='same',
                  activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.5),
    layers.Conv2D(filters=64, kernel_size=(3, 3),
                  kernel_regularizer=keras.regularizers.l2(l=0.01),
                  padding='same',
                  activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.5),
    layers.Conv2D(filters=128, kernel_size=(3, 3),
                  kernel_regularizer=keras.regularizers.l2(l=0.01),
                  padding='same',
                  activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.5),
    layers.Conv2D(filters=256, kernel_size=(3, 3),
                  kernel_regularizer=keras.regularizers.l2(l=0.01),
                  padding='same',
                  activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.5),
    layers.Conv2D(filters=512, kernel_size=(3, 3),
                  kernel_regularizer=keras.regularizers.l2(l=0.01),
                  padding='same',
                  activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10)
])

print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy']
)

model.fit(ds_train, epochs=2000, verbose=2)  # 数据预处理阶段已经定义了batch_size
model.evaluate(ds_test, verbose=2)
