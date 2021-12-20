# -*- coding:utf-8 -*-

from tensorflow.keras.layers import (Conv2D,
                                     MaxPooling2D,
                                     Dropout,
                                     Flatten,
                                     Dense,
                                     Input,
                                     Lambda)
from tensorflow.keras import Model
import tensorflow as tf

import typing

tf.config.run_functions_eagerly(True)


@tf.function
def AlexNet(input_shape: typing.Tuple[int], classes: int = 1000) -> Model:
    """
    AlexNet Model
    :param input_shape: dimension of data
    :param classes: num of target
    :return: model
    """

    x_input = Input(shape=input_shape)

    # layer1  1~5 convolution layer
    x = Conv2D(filters=95, kernel_size=(11, 11), strides=3, padding='valid', activation='relu')(x_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = Lambda(tf.nn.local_response_normalization)(x)

    # layer2
    x = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = Lambda(tf.nn.local_response_normalization)(x)

    # layer3
    x = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)

    # layer4
    x = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)

    # layer5
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = Lambda(tf.nn.local_response_normalization)(x)

    # layer6
    x = Flatten()(x)
    x = Dense(units=4096, activation='relu')(x)
    x = Dropout(rate=0.5)(x)

    # layer7
    x = Dense(units=4096, activation='relu')(x)
    x = Dropout(0.5)(x)

    # layer8
    output = Dense(units=classes, activation='softmax')(x)

    model = Model(inputs=x_input, outputs=output)
    return model
