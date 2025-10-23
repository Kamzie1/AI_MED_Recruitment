# Solution

### Data
From task_data.csv we extract usefull data. I figured we dont need xx, yy, xy since normalized vector is a value derived from these numbers. We also dont need photo id, since it is irrelevant to any disease. 

Cardiomegaly happens when patient's heart is enlarged. Generally it is a symptom of a different condition, for example heart damage. We detect Cardiomegaly by performing a chest X-ray.

That is exactly what we are given, a csv file containing data from X-ray.

Although first we need to cast strings to floats and normalize all the values.
 
Then we split data into 4 data sets. 2 for representing neighbors and 2 for testing using sklearn train_test_split function.

### KNN algorithm
We initialize knn class with training dataset. Then for each patient chosen for testing we perform KNN algorithm. Basically we calculate distances between given patient and training patients using one of the given ways: euclidean, manhattan, hamming distances between vectors. Then we take k closest patients and decide whether given patient is sick or not.

### Confusion Matrix
We evaluate three solutions that calculate distances (hamming, manhattan, euclidean). This allows us to compare these three ways of implementing KNN algorithm.

