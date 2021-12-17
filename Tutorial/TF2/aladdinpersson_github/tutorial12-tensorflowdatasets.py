# -*- coding:utf-8 -*-
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow_hub as hub
import tensorflow_datasets as tfds

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# (ds_train, ds_test), ds_info = tfds.load(
#     name="mnist",
#     split=["train", "test"],
#     shuffle_files=True,
#     as_supervised=True,
#     with_info=True
# )
#
# print(ds_info)
# Dataset mnist downloaded and prepared to C:\Users\Lis\tensorflow_datasets\mnist\1.0.0. Subsequent calls will reuse this data.
# tfds.core.DatasetInfo(
#     name='mnist',
#     version=1.0.0,
#     description='The MNIST database of handwritten digits.',
#     urls=['https://storage.googleapis.com/cvdf-datasets/mnist/'],
#     features=FeaturesDict({
#         'image': Image(shape=(28, 28, 1), dtype=tf.uint8),
#         'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=10),
#     }),
#     total_num_examples=70000,
#     splits={
#         'test': 10000,
#         'train': 60000,
#     },
#     supervised_keys=('image', 'label'),
#     citation="""@article{lecun2010mnist,
#       title={MNIST handwritten digit database},
#       author={LeCun, Yann and Cortes, Corinna and Burges, CJ},
#       journal={ATT Labs [Online]. Available: http://yann. lecun. com/exdb/mnist},
#       volume={2},
#       year={2010}
#     }""",
#     redistribution_info=,
# )
#
# import sys
# sys.exit()

BATCH_SIZE = 128
AUTOTUNE = tf.data.experimental.AUTOTUNE
# def normalized_img(image, label):
#     """ Normalizes images"""
#     return tf.cast(image, tf.float32), label
#
# ds_train = ds_train.map(normalized_img, num_parallel_call=AUTOTUNE)
# ds_train = ds_train.cache()
# ds_train = ds_train.shuffle(ds_info.split['train'].num_examples)
# ds_train = ds_train.batch(BATCH_SIZE)
# ds_train = ds_train.prefetch(AUTOTUNE)
#
# ds_test = ds_test.map(normalized_img, num_parallel_call=AUTOTUNE)
# ds_test = ds_test.batch(BATCH_SIZE)
# ds_test = ds_test.prefetch(AUTOTUNE)
#
# model = keras.Sequential([
#     keras.Input(shape=(28, 28, 1)),
#     layers.Conv2D(32, 3, activation='relu'),
#     layers.Flatten(),
#     layers.Dense(10, activation='softmax')
# ])
#
# model.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(),
#     optimizer=keras.optimizers.Adam(lr=0.01),
#     metrics=['accuracy']
# )
#
# model.fit(ds_train, epochs=5, verbose=2)
# model.evaluate(ds_test)


(ds_train, ds_test), ds_info = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,  # will return tuple (img, label) otherwise dict
    with_info=True,  # able to get info about dataset
)

tokenizer = tfds.features.text.Tokenizer()


def build_vocabulary():
    vocabulary = set()
    for text, _ in ds_train:
        vocabulary.update(tokenizer.tokenize(text.numpy().lower()))
    return vocabulary


vocabulary = build_vocabulary()

encoder = tfds.features.text.TokenTextEncoder(
    list(vocabulary), oov_token="<UNK>", lowercase=True, tokenizer=tokenizer
)


def my_enc(text_tensor, label):
    encoded_text = encoder.encode(text_tensor.numpy())
    return encoded_text, label


def encode_map_fn(text, label):
    # py_func doesn't set the shape of the returned tensors.
    encoded_text, label = tf.py_function(
        my_enc, inp=[text, label], Tout=(tf.int64, tf.int64)
    )

    # `tf.data.Datasets` work best if all components have a shape set
    #  so set the shapes manually:
    encoded_text.set_shape([None])
    label.set_shape([])

    return encoded_text, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = ds_train.map(encode_map_fn, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(1000)
ds_train = ds_train.padded_batch(32, padded_shapes=([None], ()))
ds_train = ds_train.prefetch(AUTOTUNE)

ds_test = ds_test.map(encode_map_fn)
ds_test = ds_test.padded_batch(32, padded_shapes=([None], ()))

model = keras.Sequential(
    [
        layers.Masking(mask_value=0),
        layers.Embedding(input_dim=len(vocabulary) + 2, output_dim=32),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dense(1),
    ]
)

model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(3e-4, clipnorm=1),
    metrics=["accuracy"],
)

model.fit(ds_train, epochs=15, verbose=2)
model.evaluate(ds_test)

