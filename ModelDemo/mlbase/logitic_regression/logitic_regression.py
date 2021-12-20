# -*- coding:utf8 -*-

import numpy as np
from sklearn.datasets import make_blobs


class LogisticRegression:
    def __init__(self, x, lr=0.1, max_iteration=10000):
        self.lr = lr
        self.max_iteration = max_iteration
        self.m, self.n = x.shape  # self.m: total data, self.n: data feature

    def train(self, x, y):
        self.weights = np.zeros(shape=(self.n, 1))
        self.bias = 0

        for i in range(self.max_iteration):
            y_pred = self.sigmoid(np.dot(x, self.weights) + self.bias)
            cost = -1 / self.m * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))  # 交叉熵损失

            dw = 1 / self.m * np.dot(x.T, (y_pred - y))
            db = 1 / self.m * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % 1000 == 0:
                print(f"in {i} iter with cost: {cost}")

        return self.weights, self.bias

    def predict(self, x):
        y_pred = self.sigmoid(np.dot(x, self.weights) + self.bias)
        y_pred_labels = y_pred > 0.5
        return y_pred_labels

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


if __name__ == "__main__":
    np.random.seed(1)
    X, y = make_blobs(n_samples=1000, centers=2)
    y = y[:, np.newaxis]  # m => (m, 1)

    logreg = LogisticRegression(X)
    w, b = logreg.train(X, y)
    y_predict = logreg.predict(X)

    print(f"Accuracy: {np.sum(y == y_predict) / X.shape[0]}")

