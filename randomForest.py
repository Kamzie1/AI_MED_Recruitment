from decisionTree import DecisionTreeClassifier
import numpy as np
import random


class RandomForest:
    def __init__(self, num_trees=6) -> None:
        self.num_trees = num_trees
        self.trees = []
        self.tree_features = []

    def fit(self, data):
        samples, features = data.shape
        num_features = int((features - 1) ** 0.5)
        for tree in range(self.num_trees):
            dT = DecisionTreeClassifier()
            dT.fit(self.get_random_data(data, num_features, samples, features))
            self.trees.append(dT)

    def get_random_data(self, data, num_features, samples, features):
        new_data = []
        for i in range(len(data)):
            new_data.append(data[random.randint(0, samples - 1)])

        features = [random.randint(1, features - 1) for _ in range(num_features)]
        features.insert(0, 0)
        self.tree_features.append(features)
        for idx, patient in enumerate(new_data):
            new_data[idx] = [
                value for idx, value in enumerate(patient) if idx in features
            ]

        return np.array(new_data)

    def predict(self, patient):
        predictions = []
        for features, tree in zip(self.tree_features, self.trees):
            new_patient = [
                value for idx, value in enumerate(patient) if idx in features
            ]
            predictions.append(tree.traverse_tree(np.array(new_patient)))

        return round(sum(predictions) / len(predictions))
