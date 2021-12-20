# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class KMeans:

    def __init__(self, X, num_cluster=3):
        self.k = num_cluster
        self.max_iterations = 100
        self.plot_figure = True
        self.num_examples = X.shape[0]
        self.num_features = X.shape[1]

    def initialize_random_centroids(self, X):
        centroids = np.zeros(shape=(self.k, self.num_features))

        for i in range(self.k):
            centroid = X[np.random.choice(range(self.num_examples))]  # 随机选k个点作为起始的聚类中心
            centroids[i] = centroid
        return centroids

    def create_cluster(self, X, centroids):
        clusters = [[] for _ in range(self.k)]

        for point_idx, point in enumerate(X):
            closest_centroid = np.argmin(np.sqrt(np.sum((point - centroids) ** 2, axis=1)))
            clusters[closest_centroid].append(point_idx)

        return clusters

    def calculate_new_centroids(self, clusters, X):
        centroids = np.zeros(shape=(self.k, self.num_features))
        for idx, cluster in enumerate(clusters):
            new_centroid = np.mean(X[cluster], axis=0)
            centroids[idx] = new_centroid
        return centroids

    def predict_cluster(self, clusters, X):
        y_pred = np.zeros(shape=self.num_examples)

        for cluster_idx, cluster in enumerate(clusters):
            for same_idx in cluster:
                y_pred[same_idx] = cluster_idx
        return y_pred

    def plot_fig(self, X, y):
        plt.scatter(x=X[:, 0], y=X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
        plt.show()

    def fit(self, X):
        centroids = self.initialize_random_centroids(X)
        for it in range(self.max_iterations):
            clusters = self.create_cluster(X, centroids)
            previous_centroids = centroids
            centroids = self.calculate_new_centroids(clusters, X)
            diff = centroids - previous_centroids

            if not diff.any():
                print('Termination criterion satisfied')
                break
        # Get label predictions
        y_pred = self.predict_cluster(clusters, X)
        if self.plot_figure:
            self.plot_fig(X, y_pred)
        return y_pred


if __name__ == "__main__":
    np.random.seed(10)
    num_clusters = 3
    X, _ = make_blobs(n_samples=1000, n_features=2, centers=num_clusters)
    Kmeans = KMeans(X, num_cluster=num_clusters)
    y_pred = Kmeans.fit(X)
