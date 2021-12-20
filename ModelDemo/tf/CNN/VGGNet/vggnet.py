# -*- coding:utf-8 -*-

import tensorflow as tf
import typing
import os
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Add,
    Activation,
    ZeroPadding2D,
    MaxPooling2D,
    AveragePooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.run_functions_eagerly(True)


@tf.function
def VGGNet(name: str,
           architecture: typing.List[typing.Union[int, str]],
           input_shape: typing.Tuple[int],
           classes: int = 1000) -> Model:
    X_input = Input(shape=input_shape)

    # make convolution layer
    X = make_conv_layer(X=X_input, architecture=architecture)
    # flatten the output and make fully connected layers
    X = Flatten()(X)
    X = make_dense_layer(X=X, output_units=4096)
    X = make_dense_layer(X=X, output_units=4096)
    # classifier layer
    X = Dense(units=classes, activation='softmax')(X)
    model = Model(inputs=X_input, outputs=X, name=name)
    return model


def make_conv_layer(X: tf.Tensor,
                    architecture: typing.List[typing.Union[int, str]],
                    activation: str = 'relu') -> tf.Tensor:
    """
    Method to create convolution layers for VGGNet.
    In VGGNet
        - Kernal is always 3x3 for conv-layer with padding 1 and stride 1.
        - 2x2 kernel for max pooling with stride of 2.
    Arguments:
    X            -- input tensor
    architecture -- number of output channel per convolution layers in VGGNet
    activation   -- type of activation method
    Returns:
    X           -- output tensor
    """
    for output in architecture:

        # convolution layer
        if type(output) == int:
            out_channels = output
            X = Conv2D(filters=out_channels,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       padding='same')(X)
            X = BatchNormalization()(X)
            X = Activation(activation=activation)(X)
        # max-pooling layer
        else:
            X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)
    return X


def make_dense_layer(X: tf.Tensor,
                     output_units: int,
                     dropout: int = 0.5,
                     activation='relu') -> tf.Tensor:
    """

    :param X: input Tensor
    :param output_units: output tensor size filters
    :param dropout: dropout value for regularization
    :param activation: type of activation method
    :return: tf.Tensor
    """
    X = Dense(units=output_units)(X)
    X = BatchNormalization()(X)
    X = Activation(activation=activation)(X)
    X = Dropout(rate=dropout)(X)
    return X


VGG_types = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

model = VGGNet(name="VGGNet16", architecture=VGG_types["VGG16"], input_shape=(224, 224, 3), classes=1000)
model.summary()

"""
Model: "VGGNet16"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0         
_________________________________________________________________
conv2d (Conv2D)              (None, 224, 224, 64)      1792      
_________________________________________________________________
batch_normalization (BatchNo (None, 224, 224, 64)      256       
_________________________________________________________________
activation (Activation)      (None, 224, 224, 64)      0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 224, 224, 64)      36928     
_________________________________________________________________
batch_normalization_1 (Batch (None, 224, 224, 64)      256       
_________________________________________________________________
activation_1 (Activation)    (None, 224, 224, 64)      0         
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 112, 112, 64)      0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 112, 112, 128)     73856     
_________________________________________________________________
batch_normalization_2 (Batch (None, 112, 112, 128)     512       
_________________________________________________________________
activation_2 (Activation)    (None, 112, 112, 128)     0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 112, 112, 128)     147584    
_________________________________________________________________
batch_normalization_3 (Batch (None, 112, 112, 128)     512       
_________________________________________________________________
activation_3 (Activation)    (None, 112, 112, 128)     0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 56, 56, 128)       0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 56, 56, 256)       295168    
_________________________________________________________________
batch_normalization_4 (Batch (None, 56, 56, 256)       1024      
_________________________________________________________________
activation_4 (Activation)    (None, 56, 56, 256)       0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 56, 56, 256)       590080    
_________________________________________________________________
batch_normalization_5 (Batch (None, 56, 56, 256)       1024      
_________________________________________________________________
activation_5 (Activation)    (None, 56, 56, 256)       0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 56, 56, 256)       590080    
_________________________________________________________________
batch_normalization_6 (Batch (None, 56, 56, 256)       1024      
_________________________________________________________________
activation_6 (Activation)    (None, 56, 56, 256)       0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 28, 28, 256)       0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 28, 28, 512)       1180160   
_________________________________________________________________
batch_normalization_7 (Batch (None, 28, 28, 512)       2048      
_________________________________________________________________
activation_7 (Activation)    (None, 28, 28, 512)       0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 28, 28, 512)       2359808   
_________________________________________________________________
batch_normalization_8 (Batch (None, 28, 28, 512)       2048      
_________________________________________________________________
activation_8 (Activation)    (None, 28, 28, 512)       0         
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 28, 28, 512)       2359808   
_________________________________________________________________
batch_normalization_9 (Batch (None, 28, 28, 512)       2048      
_________________________________________________________________
activation_9 (Activation)    (None, 28, 28, 512)       0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 14, 14, 512)       0         
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 14, 14, 512)       2359808   
_________________________________________________________________
batch_normalization_10 (Batc (None, 14, 14, 512)       2048      
_________________________________________________________________
activation_10 (Activation)   (None, 14, 14, 512)       0         
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 14, 14, 512)       2359808   
_________________________________________________________________
batch_normalization_11 (Batc (None, 14, 14, 512)       2048      
_________________________________________________________________
activation_11 (Activation)   (None, 14, 14, 512)       0         
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 14, 14, 512)       2359808   
_________________________________________________________________
batch_normalization_12 (Batc (None, 14, 14, 512)       2048      
_________________________________________________________________
activation_12 (Activation)   (None, 14, 14, 512)       0         
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 7, 7, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0         
_________________________________________________________________
dense (Dense)                (None, 4096)              102764544 
_________________________________________________________________
batch_normalization_13 (Batc (None, 4096)              16384     
_________________________________________________________________
activation_13 (Activation)   (None, 4096)              0         
_________________________________________________________________
dropout (Dropout)            (None, 4096)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 4096)              16781312  
_________________________________________________________________
batch_normalization_14 (Batc (None, 4096)              16384     
_________________________________________________________________
activation_14 (Activation)   (None, 4096)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 4096)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 1000)              4097000   
=================================================================
Total params: 138,407,208
Trainable params: 138,382,376
Non-trainable params: 24,832
_________________________________________________________________
"""
