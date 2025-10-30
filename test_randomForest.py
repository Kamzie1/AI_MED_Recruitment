from randomForest import RandomForest
from confMatr import ConfusionMatrix
import numpy as np

data = [[1, 8, 3], [0, 4, 4], [1, 6, 4], [0, 9, 9], [1, 8, 6]]
test_data = [[0, 5, 8], [0, 4, 9], [1, 7, 4]]

# evaluating the solution with cross validation
dT = RandomForest(6)
cfm = ConfusionMatrix()

dT.fit(np.array(data))
# calculating TP, TN, FP, FN
for point in np.array(test_data):
    cfm.evaluate(dT.predict(point), point[0])

cfm.print_all()
