# -*- coding:utf-8 -*-

from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.layers import Dense

VOCAB_SIZE = 2000
EMBEDDING_DIM = 100
MAX_WORDS = 500
CLASS_NUM = 5


def build_fasttext():
    model = Sequential()
    model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_WORDS))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(units=CLASS_NUM, activation='softmax'))
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    model = build_fasttext()
    print(model.summary())