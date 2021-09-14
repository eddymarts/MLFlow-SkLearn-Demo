from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import mlflow


# By default experiment we've set will be used
with mlflow.start_run():
# Create experiment (artifact_location=./ml_runs by default)
    mlflow.set_experiment("Dummy Experiments")
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                            test_size=0.2, shuffle=True)


    n_neighbors = 5
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
