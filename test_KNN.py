from KNN import KNN
import numpy as np


def test_manhattan():
    print("---Testing manhattan distance---")
    A = [2, 4, 4, 6]
    B = [5, 5, 7, 8]

    assert KNN.manhattan(A, B), 9


def test_eucalidan():
    print("---Testing eucalidan distance---")
    A = [25, 12, 15, 14, 19, 23, 25, 29]
    B = [5, 7, 7, 9, 12, 9, 9, 4]
    assert (
        abs(KNN.eucalidan(np.array(A), np.array(B)) - 40.496913462633174) < 0.0000000001
    )


def test_hamming():
    print("---Testing hamming distance---")
    A = [2, 4, 7, 6]
    B = [5, 5, 7, 8]

    assert KNN.hamming(A, B), 3


def test_predict():
    print("--testing knn.predict--")
    points = np.array(
        [
            [2, 4, 3],
            [1, 3, 5],
            [2, 3, 1],
            [3, 2, 3],
            [2, 1, 6],
            [5, 6, 5],
            [4, 5, 2],
            [4, 6, 1],
            [6, 6, 1],
            [5, 4, 6],
            [10, 10, 4],
        ]
    )

    solutions = np.array([[0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1]])
    knn = KNN(points, solutions)

    new_point = [3, 3, 2]

    assert round(knn.predict(new_point)) == 0


def run_tests():
    test_manhattan()
    test_eucalidan()
    test_hamming()
    test_predict()


if __name__ == "__main__":
    run_tests()
