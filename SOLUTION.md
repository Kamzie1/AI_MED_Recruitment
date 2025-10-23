# Solution

### Data
From task_data.csv we extract usefull data. I figured we dont need xx, yy, xy since normalized vector is a value derived from these numbers. We also dont need photo id, since it is irrelevant to any disease. Then we need to cast strings to floats and normalize all the values.

Cardiomegaly happens when patient's heart is enlarged. Generally it is a symptom of a different condition, for example heart damage. We detect Cardiomegaly by performing a chest X-ray.
 
### KNN algorithm
Using cross-validation we initialize knn class with training dataset. Then for each patient chosen for testing we perform KNN algorithm. Basically we calculate distances between given patient and training patients using one of the given methods: euclidean, manhattan, hamming. Then we take k closest patients and decide whether given patient is sick or not.

### Confusion Matrix
We compare three methods of calculating distances (hamming, manhattan, euclidean)

