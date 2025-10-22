class ConfusionMatrix:
    def __init__(self) -> None:
        self.TP: int = 0
        self.TN: int = 0
        self.FP: int = 0
        self.FN: int = 0

    def evaluate(self, prediction: int, solution: int) -> None:
        if prediction == solution:
            if solution == 1:
                self.TP += 1
            else:
                self.TN += 1
        else:
            if solution == 1:
                self.FP += 1
            else:
                self.FN += 1

    @property
    def accuracy(self) -> float:
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)

    @property
    def precision(self) -> float:
        return self.TP / (self.TP + self.FP)

    @property
    def recall(self) -> float:
        return self.TP / (self.TP + self.FN)

    @property
    def F1_score(self) -> float:
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    def print_all(self) -> None:
        print(
            f"Accuracy: {self.accuracy} Precision: {self.precision} Recall: {self.recall} F1-score: {self.F1_score}"
        )
