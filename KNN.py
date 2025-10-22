import numpy as np
import numpy.typing as npt
from collections.abc import Callable


class KNN:
    def __init__(self, points: npt.NDArray, solutions: npt.NDArray, k: int = 3) -> None:
        self.k: int = 3
        self.points: npt.NDArray = points
        self.solutions: npt.NDArray = solutions

    @staticmethod
    def manhattan(a: npt.NDArray, b: npt.NDArray) -> float:
        return sum(abs(v1 - v2) for v1, v2 in zip(a, b))

    @staticmethod
    def euclidean(a: npt.NDArray, b: npt.NDArray) -> float:
        return np.sum((a - b) ** 2)

    @staticmethod
    def hamming(a: npt.NDArray, b: npt.NDArray) -> float:
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
        distances = []

        for point, solution in zip(self.points, self.solutions):
            distances.append((solution, calc_dist(point, new_point)))

        distances = sorted(distances, key=lambda x: x[1])
        suma = 0
        for i in range(self.k):
            suma += distances[i][0]

        return suma / self.k
