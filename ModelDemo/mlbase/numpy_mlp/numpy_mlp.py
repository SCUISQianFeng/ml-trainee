#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuisqianfeng 
@license: Apache Licence 
@file: numpy_mlp.py 
@time: 2022/01/24
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""

import numpy as np
import sklearn
from sklearn import datasets
from sklearn import linear_model

# hyper parameters
seed = 1
np.random.seed(seed)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_data():
    return datasets.make_gaussian_quantiles(mean=None, cov=0.7, n_samples=200, n_features=2, n_classes=2, shuffle=True,
                                            random_state=seed)


gaussian_quantiles = load_data()
X, Y = gaussian_quantiles

clf = linear_model.LogisticRegressionCV()
clf.fit(X, Y)
pred = clf.predict(X)
print("LR Accuracy: ", np.sum(pred == Y) / Y.shape[0])


# lr_score = clf.score(X, Y)
# print(lr_score)
class MLP(object):

    def __init__(self, input_dim: int = 2, hidden_size: int = 2, output_dim: int = 1, epochs: int = 5000,
                 lr: float = 1.2):
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.epochs = epochs
        self.lr = lr
        self.W1 = np.random.randn(self.hidden_size, self.input_dim) * 0.01  # 10, 2
        self.b1 = np.zeros(shape=(self.hidden_size, 1))
        self.W2 = np.random.randn(self.output_dim, self.hidden_size) * 0.01  # 1, 10
        self.b2 = np.zeros(shape=(self.output_dim, 1))

    def forward(self, X):
        self.Z1 = self.W1.dot(X.T) + self.b1  # [200, 2] => [10, 200]
        self.A1 = sigmoid(self.Z1)
        self.Z2 = self.W2.dot(self.A1) + self.b2  # [10, 200] => [1, 200]
        self.A2 = sigmoid(self.Z2)

    def back_prop(self, X, Y):
        m = X.shape[0]
        self.dZ2 = self.A2 - Y
        self.dW2 = (1 / m) * np.dot(self.dZ2, self.A1.T)  # 1, 10
        self.db2 = (1 / m) * np.sum(self.dZ2, axis=1, keepdims=True)  # 1, 1
        self.dZ1 = np.multiply(np.dot(self.W2.T, self.dZ2), np.multiply(self.A1, 1 - self.A1))
        self.dW1 = (1 / m) * np.dot(self.dZ1, X)
        self.db1 = (1 / m) * np.sum(self.dZ1, axis=1, keepdims=True)

    def train(self, X, Y):
        m = X.shape[0]
        for epoch in range(self.epochs):
            self.forward(X)
            self.back_prop(X, Y)
            self.W1 -= self.lr * self.dW1
            self.b1 -= self.lr * self.db1
            self.W2 -= self.lr * self.dW2
            self.b2 -= self.lr * self.db2

            if epoch % 1000 == 0:
                # 交叉熵
                loss = -np.sum(np.multiply(np.log(self.A2), Y) + np.multiply(np.log(1 - self.A2), 1 - Y)) / m
                print('Loss ', epoch, ' = ', loss)

    def predict(self, X):
        self.forward(X)
        return np.round(self.A2).astype(int)


nn = MLP(2, 10, 1, 5000, 1.2)
nn.train(X, Y)
nn_predictions = nn.predict(X)
print("Neural Network accuracy : ", np.sum(nn_predictions == Y) / Y.shape[0])
