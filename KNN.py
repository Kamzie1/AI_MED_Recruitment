import numpy as np
import numpy.typing as npt
from collections.abc import Callable


class KNN:
    def __init__(self, points: npt.NDArray, k: int = 3) -> None:
        self.k: int = 3
        self.points: npt.NDArray = points

    @staticmethod
    def manhattan(a: npt.NDArray, b: npt.NDArray) -> float:
        """
        calculates manhattan distance between vectors a and b.
        """
        return sum(abs(v1 - v2) for v1, v2 in zip(a, b))

    @staticmethod
    def euclidean(a: npt.NDArray, b: npt.NDArray) -> float:
        """
        calculates eucalidean distance between vectors a and b.
        """
        return np.sum((a - b) ** 2)

    @staticmethod
    def hamming(a: npt.NDArray, b: npt.NDArray) -> float:
        """
        calculates hamming distance between vectors a and b.
        """
        diff = 0
        for v1, v2 in zip(a, b):
            if v1 != v2:
                diff += 1
        return diff

    def predict(
        self,
        new_point: npt.NDArray,
        calc_dist: Callable[[npt.NDArray, npt.NDArray], float] = euclidean,
    ) -> float:
        """
        calculates distance between all the points from x_train and based on k-closest predicts wheter new_point is sick or not.
        """
        distances = []

        for point in self.points:
            distances.append((point[0], calc_dist(point[1:], new_point)))

        distances = sorted(distances, key=lambda x: x[1])
        suma = 0
        for i in range(self.k):
            suma += distances[i][0]

        return suma / self.k
