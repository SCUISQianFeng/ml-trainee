# -*- coding:utf-8 -*-

import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    MaxPooling2D,
    Dropout,
    Flatten,
    BatchNormalization,
    AveragePooling2D,
    concatenate,
    Activation,
)
import typing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.run_functions_eagerly(True)

from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    concatenate,
)


@tf.function
def convolution_block(
        X: tf.Tensor,
        filters: int,
        kernel_size: int,
        stride: int = 1,
        padding: str = 'valid',
) -> tf.Tensor:
    """
    Convolution block for GoogLeNet.
    Arguments:
    X           -- input tensor of shape (m, H, W, filters)
    filters      -- defining the number of filters in the CONV layers
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    stride      -- integer specifying the stride to be used
    padding     -- padding type, same or valid. Default is valid
    Returns:
    X           -- output of the identity block, tensor of shape (H, W, filters)
    """

    X = Conv2D(
        filters=filters,
        kernel_size=(kernel_size, kernel_size),
        strides=(stride, stride),
        padding=padding,
    )(X)
    # batch normalization is not in original paper because it was not invented at that time
    # however I am using it here because it will improve the performance
    X = BatchNormalization()(X)
    X = Activation("relu")(X)

    return X


@tf.function
def inception_block(
        X: tf.Tensor,
        filters_1x1: int,
        filters_3x3_reduce: int,
        filters_3x3: int,
        filters_5x5_reduce: int,
        filters_5x5: int,
        pool_size: int,
) -> tf.Tensor:
    """
    Inception block for GoogLeNet.
    Arguments:
    X                  -- input tensor of shape (m, H, W, filters)
    filters_1x1        -- number of filters for (1x1 conv) in first branch
    filters_3x3_reduce -- number of filters for (1x1 conv) dimensionality reduction before (3x3 conv) in second branch
    filters_3x3        -- number of filters for (3x3 conv) in second branch
    filters_5x5_reduce -- number of filters for (1x1 conv) dimensionality reduction before (5x5 conv) in third branch
    filters_5x5        -- number of filters for (5x5 conv) in third branch
    pool_size          -- number of filters for (1x1 conv) after 3x3 max pooling in fourth branch
    Returns:
    X                  -- output of the identity block, tensor of shape (H, W, filters)
    """

    # first branch
    conv_1x1 = convolution_block(
        X,
        filters=filters_1x1,
        kernel_size=1,
        padding="same"
    )

    # second branch
    conv_3x3 = convolution_block(
        X,
        filters=filters_3x3_reduce,
        kernel_size=1,
        padding="same"
    )
    conv_3x3 = convolution_block(
        conv_3x3,
        filters=filters_3x3,
        kernel_size=3,
        padding="same"
    )

    # third branch
    conv_5x5 = convolution_block(
        X,
        filters=filters_5x5_reduce,
        kernel_size=1,
        padding="same"
    )
    conv_5x5 = convolution_block(
        conv_5x5,
        filters=filters_5x5,
        kernel_size=5,
        padding="same"
    )

    # fourth branch
    pool_projection = MaxPooling2D(
        pool_size=(2, 2),
        strides=(1, 1),
        padding="same",
    )(X)
    pool_projection = convolution_block(
        pool_projection,
        filters=pool_size,
        kernel_size=1,
        padding="same"
    )

    # concat by channel/filter
    return concatenate(inputs=[conv_1x1, conv_3x3, conv_5x5, pool_projection], axis=3)


@tf.function
def auxiliary_block(
        X: tf.Tensor,
        classes: int,
) -> tf.Tensor:
    """
    Auxiliary block for GoogLeNet.
    Refer to the original paper, page 8 for the auxiliary layer specification.
    Arguments:
    X       -- input tensor of shape (m, H, W, filters)
    classes -- number of classes for classification
    Return:
    X       -- output of the identity block, tensor of shape (H, W, filters)
    """

    X = AveragePooling2D(
        pool_size=(5, 5),
        padding="same",
        strides=(3, 3),
    )(X)
    X = convolution_block(
        X,
        filters=128,
        kernel_size=1,
        stride=1,
        padding="same",
    )
    X = Flatten()(X)
    X = Dense(units=1024, activation="relu")(X)
    X = Dropout(rate=0.7)(X)
    X = Dense(units=classes)(X)
    X = Activation("softmax")(X)

    return X


# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# goog_le_net_block (GoogLeNet (None, 112, 28, 32)       34944


@tf.function
def GoogLeNet(input_shape: typing.Tuple[int] = (224, 224, 3), classes: int = 1000) -> Model:
    """
    Implementation of the popular GoogLeNet aka Inception v1 architecture.
    Refer to the original paper, page 6 - table 1 for inception block filter sizes.
    Arguments:
    input_shape -- shape of the images of the dataset
    classes     -- number of classes for classification
    Returns:
    model       -- a Model() instance in Keras
    """

    # convert input shape into tensor
    X_input = Input(input_shape)

    # NOTE: auxiliary layers are only used in trainig phase to improve performance
    #       because they act as regularization and prevent vanishing gradient problem
    auxiliary1 = None  # to store auxiliary layers classification value
    auxiliary2 = None

    # layer 1 (convolution block)
    X = convolution_block(
        X=X_input,
        filters=64,
        kernel_size=7,
        stride=2,
        padding="same",
    )

    # layer 2 (max pool)
    X = MaxPooling2D(
        pool_size=(3, 3),
        padding="same",
        strides=(2, 2),
    )(X)

    # layer 3 (convolution block)
    # 1x1 reduce
    X = convolution_block(
        X,
        filters=64,
        kernel_size=1,
        stride=1,
        padding="same",
    )
    X = convolution_block(
        X,
        filters=192,
        kernel_size=3,
        stride=1,
        padding="same",
    )

    # layer 4 (max pool)
    X = MaxPooling2D(
        pool_size=(3, 3),
        padding="same",
        strides=(2, 2),
    )(X)

    # layer 5 (inception 3a)
    X = inception_block(
        X,
        filters_1x1=64,
        filters_3x3_reduce=96,
        filters_3x3=128,
        filters_5x5_reduce=16,
        filters_5x5=32,
        pool_size=32,
    )

    # layer 6 (inception 3b)
    X = inception_block(
        X,
        filters_1x1=128,
        filters_3x3_reduce=128,
        filters_3x3=192,
        filters_5x5_reduce=32,
        filters_5x5=96,
        pool_size=64,
    )

    # layer 7 (max pool)
    X = MaxPooling2D(
        pool_size=(3, 3),
        padding="same",
        strides=(2, 2),
    )(X)

    # layer 8 (inception 4a)
    X = inception_block(
        X,
        filters_1x1=192,
        filters_3x3_reduce=96,
        filters_3x3=208,
        filters_5x5_reduce=16,
        filters_5x5=48,
        pool_size=64,
    )

    # First Auxiliary Softmax Classifier
    auxiliary1 = auxiliary_block(X, classes=classes)

    # layer 9 (inception 4b)
    X = inception_block(
        X,
        filters_1x1=160,
        filters_3x3_reduce=112,
        filters_3x3=224,
        filters_5x5_reduce=24,
        filters_5x5=64,
        pool_size=64,
    )

    # layer 10 (inception 4c)
    X = inception_block(
        X,
        filters_1x1=128,
        filters_3x3_reduce=128,
        filters_3x3=256,
        filters_5x5_reduce=24,
        filters_5x5=64,
        pool_size=64,
    )

    # layer 11 (inception 4d)
    X = inception_block(
        X,
        filters_1x1=112,
        filters_3x3_reduce=144,
        filters_3x3=288,
        filters_5x5_reduce=32,
        filters_5x5=64,
        pool_size=64,
    )

    # Second Auxiliary Softmax Classifier
    auxiliary2 = auxiliary_block(X, classes=classes)

    # layer 12 (inception 4e)
    X = inception_block(
        X,
        filters_1x1=256,
        filters_3x3_reduce=160,
        filters_3x3=320,
        filters_5x5_reduce=32,
        filters_5x5=128,
        pool_size=128,
    )

    # layer 13 (max pool)
    X = MaxPooling2D(
        pool_size=(3, 3),
        padding="same",
        strides=(2, 2),
    )(X)

    # layer 14 (inception 5a)
    X = inception_block(
        X,
        filters_1x1=256,
        filters_3x3_reduce=160,
        filters_3x3=320,
        filters_5x5_reduce=32,
        filters_5x5=128,
        pool_size=128,
    )

    # layer 15 (inception 5b)
    X = inception_block(
        X,
        filters_1x1=384,
        filters_3x3_reduce=192,
        filters_3x3=384,
        filters_5x5_reduce=48,
        filters_5x5=128,
        pool_size=128,
    )

    # layer 16 (average pool)
    X = AveragePooling2D(
        pool_size=(7, 7),
        padding="same",
        strides=(1, 1),
    )(X)

    # layer 17 (dropout 40%)
    X = Dropout(rate=0.4)(X)

    # layer 18 (fully-connected layer with softmax activation)
    X = Dense(units=classes, activation='softmax')(X)

    model = Model(X_input, outputs=[X, auxiliary1, auxiliary2], name='GoogLeNet/Inception-v1')
    return model


model = GoogLeNet(input_shape=(224, 224, 3), classes=1000)

print(model.summary())
