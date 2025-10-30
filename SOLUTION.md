# Solution

### Data
From task_data.csv we extract usefull data. I figured we dont need xx, yy, xy since normalized vector is a value derived from these numbers. We also dont need photo id, since it is irrelevant to any disease. Then we need to cast strings to floats and normalize all the values.

Cardiomegaly happens when patient's heart is enlarged. Generally it is a symptom of a different condition, for example heart damage. We detect Cardiomegaly by performing a chest X-ray.
 
### KNN algorithm
Using cross-validation we initialize knn class with training dataset. Then for each patient chosen for testing we perform KNN algorithm. Basically we calculate distances between given patient and training patients using one of the given methods: euclidean, manhattan, hamming, chebyshev. Then we take k closest patients and decide whether given patient is sick or not.

### Decision Tree
We split the data recursively until we get only leafs or we reach maximum depth. Leafs are nodes with only one type of state(sick or healthy). We chose the best treshold and feature of split by calculating for each sample and feature gini impurity score and chosing the highest one.

### Random Forest
Random Forest choses randomizes samples for each of its trees and then compares the evaluation of each tree to give a final prediction. Besides randomizing sample it also randomizes features. The best amount of features used by each tree, based on research, is square root of the original amount of features. By using each tree with randomized samples it minimizes possible bias of a single tree. Based on this research https://www.researchgate.net/publication/230766603_How_Many_Trees_in_a_Random_Forest 64 to 128 is the best amount of trees in a random forest. Since we have a really small dataset I used only 64.

### Confusion Matrix
We count TP, TN, FP, FN by comparing models prediction and actual state of a patient, and then we calculate accuracy, precision, recall, F1 score.

