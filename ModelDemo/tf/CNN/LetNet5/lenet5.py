# -*- coding:utf-8 -*-

import tensorflow as tf
import os
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (

    Conv2D,
    MaxPooling2D,
    Dense,
    Activation,
    BatchNormalization,
    Flatten,
    AveragePooling2D
)
import typing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.run_functions_eagerly(True)


@tf.function
def lenet5(input_shape: typing.Tuple = (32, 32, 1), classes: int = 10) -> Model:
    x_input = Input(shape=input_shape)

    # layer1 C1
    x = Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='tanh')(x_input)
    # layer2 S2
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    # layer3 C3
    x = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='tanh')(x)
    # layer4 S4
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    # layer5 C5
    x = Conv2D(filters=120, kernel_size=(5, 5), strides=(1, 1), activation='tanh')(x)
    # layer6 C6
    x = Flatten()(x)
    x = Dense(units=84, activation='tanh')(x)
    # layer7 output
    outputs = Dense(units=classes, activation='softmax')(x)

    model = Model(inputs=x_input, outputs=outputs, name='LeNet5')
    return model


model = lenet5(input_shape=(32, 32, 1), classes=10)
print(model.summary())

# Model: "LeNet5"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 32, 32, 1)]       0
# _________________________________________________________________
# conv2d (Conv2D)              (None, 28, 28, 6)         156
# _________________________________________________________________
# average_pooling2d (AveragePo (None, 14, 14, 6)         0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 10, 10, 16)        2416
# _________________________________________________________________
# average_pooling2d_1 (Average (None, 5, 5, 16)          0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 1, 1, 120)         48120
# _________________________________________________________________
# flatten (Flatten)            (None, 120)               0
# _________________________________________________________________
# dense (Dense)                (None, 84)                10164
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                850
# =================================================================
# Total params: 61,706
# Trainable params: 61,706
# Non-trainable params: 0
# _________________________________________________________________
# None
