import numpy as np
from collections import Counter
import math


class Node:
    def __init__(
        self,
        left=None,
        right=None,
        treshold=None,
        feature=None,
        info_gain=None,
        value=None,
    ) -> None:
        self.left = left
        self.right = right
        self.treshold = treshold
        self.feature = feature
        self.info_gain = info_gain
        self.value = value


class DecisionTreeClassifier:
    def __init__(self, max_depth=6, min_sample=1) -> None:
        self.root = None
        self.max_depth = max_depth
        self.min_sample = 1

    def fit(self, data):
        self.root = self.craftTree(data)

    def is_leaf(self, Y):
        i = Y[0]
        for val in Y:
            if i != val:
                return False
        return True

    def craftTree(self, data, depth=0):
        Y = [point[0] for point in data]
        if depth > self.max_depth or self.min_sample > len(data) or self.is_leaf(Y):
            if len(data) == 0:
                return None
            return Node(value=self.most_common_label(Y))

        best_value = self.get_best_value(data)
        left = self.craftTree(best_value["left_data"], depth + 1)
        right = self.craftTree(best_value["right_data"], depth + 1)

        return Node(
            left,
            right,
            best_value["treshold"],
            best_value["feature"],
        )

    def most_common_label(self, Y):
        return round(sum(Y) / len(Y))

    def get_best_value(self, data) -> dict:
        samples, features = data.shape
        max_score = 0
        max_treshold = (0, 1)
        sick = 0
        healthy = 0
        for patient in data:
            if patient[0] == 0:
                healthy += 1
            else:
                sick += 1
        Gini_before_split = self.Gini_Impurity(sick, healthy)
        for feature in range(1, features):
            for sample in range(samples):
                treshold = data[sample][feature]
                left0 = 0
                left1 = 0
                right0 = 0
                right1 = 0
                for patient in data:
                    if patient[feature] <= treshold:
                        if patient[0] == 0:
                            left0 += 1
                        else:
                            left1 += 1
                    else:
                        if patient[0] == 0:
                            right0 += 1
                        else:
                            right1 += 1
                if left1 + left0 == 0 or right1 + right0 == 0:
                    continue
                score = self.info_gain(
                    Gini_before_split,
                    self.Gini_Impurity(left0, left1),
                    self.Gini_Impurity(right0, right1),
                )
                if score >= max_score:
                    max_score = score
                    max_treshold = (sample, feature)

        sample, feature = max_treshold
        treshold = data[sample][feature]
        right_data = []
        left_data = []
        for patient in data:
            if patient[feature] <= treshold:
                left_data.append(patient)
            else:
                right_data.append(patient)

        return {
            "treshold": treshold,
            "right_data": np.array(right_data),
            "left_data": np.array(left_data),
            "feature": feature,
        }

    def info_gain(self, G, G1, G2):
        return G - ((G1 + G2) / 2)

    def Gini_Impurity(self, y1, y2):
        return 1 - (y1 / (y1 + y2)) ** 2 - (y2 / (y1 + y2)) ** 2

    def traverse_tree(self, patient) -> int:
        node = self.root
        if node is None:
            print("root is None!")
            return 0

        while node.value is None:
            if patient[node.feature] <= node.treshold:
                node = node.left
            else:
                node = node.right

        return node.value
