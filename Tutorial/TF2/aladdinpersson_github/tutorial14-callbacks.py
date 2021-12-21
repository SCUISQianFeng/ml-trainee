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
    name='mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32


def normalize_img(image, label):
    """Normalizes images"""
    return tf.cast(image, tf.float32) / 255.0, label


ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.prefetch(AUTOTUNE)

save_back = keras.callbacks.ModelCheckpoint(filepath='checkpoint/',
                                            monitor='train_acc',
                                            save_weights_only=True,
                                            save_best_only=False)

lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                 factor=0.1, patience=3,
                                                 mode='max',
                                                 verbose=1)


class MyCallBack(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 1:
            print('Accuracy over 70%, quitting training')
            self.model.stop_training = True


model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
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
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10)
])

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(lr=0.01),
              metrics=['accuracy'])
model.fit(ds_train, epochs=100, verbose=2, callbacks=[save_back, lr_scheduler, MyCallBack()])
model.evaluate(ds_test, verbose=2)