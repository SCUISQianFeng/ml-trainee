import numpy as np
from collections import Counter

import sys

sys.path.append('..')


class DecisionTree:
    def __init__(self, max_depth, min_node_size):
        self.max_depth = max_depth
        self.min_node_size = min_node_size
        self.final_tree = {}

    def calculate_gini(self, child_nodes):
        """
        child nodes
        :param child_nodes: [[left_nodes], [right_nodes]]
        :return:
        """
        n = 0
        for node in child_nodes:
            n = n + len(node)  #

        gini = 0
        for node in child_nodes:
            m = len(node)  # 某一切分数组的总数
            if m == 0:
                continue
            y = []
            for row in node:
                y.append(row[-1])  # target分布

            freq = Counter(y).values()
            node_gini = 1
            for i in freq:
                node_gini = node_gini - (i / m) ** 2  # 1 - p**2
            gini = gini + (m / n) * node_gini  # pD * g(D, Ai)
        return gini

    def apply_split(self, feature_index, threshold, data):
        instances = data.tolist()
        left_child = []
        right_child = []
        for row in instances:
            if row[feature_index] < threshold:
                left_child.append(row)
            else:
                right_child.append(row)
        left_child = np.array(left_child)
        right_child = np.array(right_child)
        return left_child, right_child

    def find_best_split(self, data):
        num_of_features = len(data[0]) - 1
        gini_scores = 1000
        feature_index = 0
        feature_value = 0

        for column in range(num_of_features):
            for row in data:
                value = row[column]
                left, right = self.apply_split(column, value, data)
                children = [left, right]
                scores = self.calculate_gini(child_nodes=children)
                if scores < gini_scores:
                    feature_index = column
                    feature_value = value
                    gini_scores = scores
                    child_nodes = children

        node = {"feature": feature_index, "value": feature_value, "children": child_nodes}
        return node

    def calc_class(self, node):
        y = []
        for row in node:
            y.append(row[-1])
        count_y = Counter(y)
        return count_y.most_common(1)[0][0]

    def recursive_split(self, node, depth):
        l, r = node["children"]
        del node["children"]
        if l.size == 0:
            c_value = self.calc_class(r)
            node["left"] = node["right"] = {"class_value": c_value, "depth": depth}
            return
        elif r.size == 0:
            c_value = self.calc_class(l)
            node["left"] = node["right"] = {"class_value": c_value, "depth": depth}
            return
        # Check if tree has reached max depth
        if depth >= self.max_depth:
            # Terminate left child node
            c_value = self.calc_class(l)
            node["left"] = {"class_value": c_value, "depth": depth}
            # Terminate right child node
            c_value = self.calc_class(r)
            node["right"] = {"class_value": c_value, "depth": depth}
            return
        # process left child
        if len(l) <= self.min_node_size:
            c_value = self.calc_class(l)
            node["left"] = {"class_value": c_value, "depth": depth}
        else:
            node["left"] = self.find_best_split(l)
            self.recursive_split(node["left"], depth + 1)
        # process right child
        if len(r) <= self.min_node_size:
            c_value = self.calc_class(r)
            node["right"] = {"class_value": c_value, "depth": depth}
        else:
            node["right"] = self.find_best_split(r)
            self.recursive_split(node["right"], depth + 1)

    """
        Apply the recursive split algorithm on the data in order to build the decision tree
        Parameters:
        X (np.array): Training data

        Returns tree (dict): The decision tree in the form of a dictionary.
    """

    def train(self, X):
        # Create initial node
        tree = self.find_best_split(X)
        # Generate the rest of the tree via recursion
        self.recursive_split(tree, 1)
        self.final_tree = tree
        return tree

    """
        Prints out the decision tree.
        Parameters:
        tree (dict): Decision tree

    """

    def print_dt(self, tree, depth=0):
        if "feature" in tree:
            print(
                "\nSPLIT NODE: feature #{} < {} depth:{}\n".format(
                    tree["feature"], tree["value"], depth
                )
            )
            self.print_dt(tree["left"], depth + 1)
            self.print_dt(tree["right"], depth + 1)
        else:
            print(
                "TERMINAL NODE: class value:{} depth:{}".format(
                    tree["class_value"], tree["depth"]
                )
            )

    def predict_single(self, tree, instance):
        if not tree:
            print("ERROR: Please train the decision tree first")
            return -1
        if "feature" in tree:
            if instance[tree["feature"]] < tree["value"]:
                return self.predict_single(tree["left"], instance)
            else:
                return self.predict_single(tree["right"], instance)
        else:
            return tree["class_value"]

    def predict(self, X):
        y_predict = []
        for row in X:
            y_predict.append(self.predict_single(self.final_tree, row))
        return np.array(y_predict)


if __name__ == "__main__":
    # # test dataset
    # X = np.array([[1, 1,0], [3, 1, 0], [1, 4, 0], [2, 4, 1], [3, 3, 1], [5, 1, 1]])
    # y = np.array([0, 0, 0, 1, 1, 1])

    train_data = np.loadtxt("example_data/data.txt", delimiter=",")
    train_y = np.loadtxt("example_data/targets.txt")

    # Build tree
    dt = DecisionTree(5, 1)
    tree = dt.train(train_data)
    y_pred = dt.predict(train_data)
    print(f"Accuracy: {sum(y_pred == train_y) / train_y.shape[0]}")
