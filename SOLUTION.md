# Solution

### Data
From task_data.csv we extract usefull data. I figured we dont need xx, yy, xy since normalized vector is a value derived from these numbers. We also dont need photo id, since it is irrelevant to any disease. 

Cardiomegaly happens when patient's heart is enlarged. Generally it is a symptom of a different condition, for example heart damage. We detect Cardiomegaly by performing a chest X-ray.

That is exactly what we are given, a csv file containing data from X-ray.

