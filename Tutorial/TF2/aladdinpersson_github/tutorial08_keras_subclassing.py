# -*- coding:utf-8 -*-
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape([-1, 28, 28, 1]).astype('float32') / 255.
x_test = x_test.reshape([-1, 28, 28, 1]).astype('float32') / 255.


class CNNBlock(keras.layers.Layer):

    def __init__(self, out_channels, kernel_size=3):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size=kernel_size, padding='same')
        self.bn = layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x

class ResBlock(layers.Layer):

    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = CNNBlock(channels[0], 3)
        self.conv2 = CNNBlock(channels[1], 3)
        self.conv3 = CNNBlock(channels[2], 3)
        self.pooling = layers.MaxPooling2D()
        self.indentify_mapping = layers.Conv2D(channels[1], 3, padding='same')

    def call(self, inputs_tensor, training=False):
        x = self.conv1(inputs_tensor, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x + self.indentify_mapping(inputs_tensor), training=training)
        x = self.pooling(x)
        return x

class ResNet_Like(keras.Model):
    def __init__(self, num_classes=10):
        super(ResNet_Like, self).__init__()
        self.block1 = ResBlock([32, 32, 64])
        self.block2 = ResBlock([128, 128, 256])
        self.block3 = ResBlock([128, 256, 512])
        self.pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(num_classes)

    def call(self, input_tensor, training=False, **kwargs):
        x = self.block1(input_tensor, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.pool(x, training=training)
        x = self.classifier(x)
        return x

    def model(self):
        x = keras.Input(shape=(28, 28, 1))
        return keras.Model(inputs=[x], outputs=self.call(x, ))



# model = keras.Sequential([
#     keras.Input(shape=(28, 28 ,1)),
#     CNNBlock(32),
#     CNNBlock(64),
#     CNNBlock(128),
#     layers.Flatten(),
#     layers.Dense(10)])
model = ResNet_Like().model()

base_input = model.layers[0].input
base_output= model.layers[-1].output
output = layers.Dense(10)(layers.Flatten()(base_output))
model = keras.Model(base_input, output)

print(model.summary())

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=64, epochs=1, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)
model.save("pretrained")
