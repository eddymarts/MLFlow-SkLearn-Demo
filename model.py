from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import mlflow
import argparse

# mlflow.set_tracking_uri("http://localhost:5000")

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                        test_size=0.2, shuffle=True)

# THIS IS HOW YOU PARSE ARGUMENTS FROM THE COMMAND LINE
def get_flags_passed_in_from_terminal():
	parser = argparse.ArgumentParser()
	parser.add_argument('-r')
	args = parser.parse_args()
	return args
args = get_flags_passed_in_from_terminal()
number =  int(args.r)
print("Arg parsed for number of neighours:", type(number), number)

# By default experiment we've set will be used
with mlflow.start_run():
# Create experiment (artifact_location=./ml_runs by default)
    mlflow.set_experiment("MLFlow-SkLearn-Demo")
    knn = KNeighborsClassifier(n_neighbors=number)
    knn.fit(X_train, y_train)

    mlflow.sklearn.log_model(sk_model=knn,
    artifact_path='sklearn_model',
    registered_model_name='sklearn-knn-classification')

    score = knn.score(X_test, y_test)

    mlflow.log_metric("mean_accuracy", score)
