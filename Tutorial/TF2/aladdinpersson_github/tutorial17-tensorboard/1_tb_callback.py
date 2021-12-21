# -*- coding:utf-8 -*-

import os

import tensorflow as tf
import tensorboard as tb
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# load data
(ds_train, ds_test), ds_info = tfds.load(
    "cifar10",
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# data preprocessing
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64


def normalize_img(image, label):
    # 统一出来成float32格式
    return tf.cast(image, tf.float32), label


def augment(image, label):
    """数据增强"""
    # 尺寸统一
    new_height = new_width = 32
    image = tf.image.resize(images=image, size=(new_width, new_height))

    # 随机rgb -> gray
    if tf.random.uniform((), minval=0, maxval=1) < 0.1:
        image = tf.tile(input=tf.image.rgb_to_grayscale(image), multiples=[1, 1, 3])
    # 随机亮度
    image = tf.image.random_brightness(image, max_delta=0.1, seed=2021)
    # 随机对比度
    image = tf.image.random_contrast(image, lower=0.1, upper=0.2, seed=2021)
    # 随机镜像反转
    image = tf.image.random_flip_left_right(image, seed=2021)
    return image, label


# pipeline preprocessing
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.map(augment, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.map(augment, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.prefetch(AUTOTUNE)

# build model
model = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(64, (3, 3), padding="same"),
    layers.ReLU(),
    layers.Conv2D(128, (3, 3), padding="same"),
    layers.ReLU(),
    layers.Flatten(),
    layers.Dense(10),
])
print(model.summary())
tb_callback = keras.callbacks.TensorBoard(log_dir='log_callback', histogram_freq=1, update_freq='epoch')

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(lr=0.002),
              metrics=['accuracy'])
model.fit(ds_train, epochs=100, verbose=2, callbacks=[tb_callback])
model.evaluate(ds_test, verbose=2)
