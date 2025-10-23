from KNN import KNN
import numpy as np


def test_manhattan():
    print("---Testing manhattan distance---")
    A = [2, 4, 4, 6]
    B = [5, 5, 7, 8]

    assert KNN.manhattan(np.array(A), np.array(B)), 9


def test_eucalidan():
    print("---Testing eucalidan distance---")
    A = [2, 2]
    B = [2, 2]
    assert abs(KNN.euclidean(np.array(A), np.array(B)) - 0) < 0.0000000001


def test_hamming():
    print("---Testing hamming distance---")
    A = [2, 4, 7, 6]
    B = [5, 5, 7, 8]

    assert KNN.hamming(np.array(A), np.array(B)), 3


def test_predict():
    print("--testing knn.predict--")
    points = np.array(
        [
            [0, 2, 4, 3],
            [0, 1, 3, 5],
            [0, 2, 3, 1],
            [0, 3, 2, 3],
            [0, 2, 1, 6],
            [0, 5, 6, 5],
            [1, 4, 5, 2],
            [1, 4, 6, 1],
            [1, 6, 6, 1],
            [1, 5, 4, 6],
            [1, 10, 10, 4],
        ]
    )

    knn = KNN(points)

    new_point = [3, 3, 2]

    assert np.round(knn.predict(np.array(new_point))) == 0


def run_tests():
    test_manhattan()
    test_eucalidan()
    test_hamming()
    test_predict()


if __name__ == "__main__":
    run_tests()
