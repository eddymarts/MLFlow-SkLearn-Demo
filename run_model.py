from sklearn import datasets
from sklearn.metrics import f1_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mlflow

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                        test_size=0.2, shuffle=True)

sc = StandardScaler().fit(X_train)
X_train =  sc.transform(X_train)
X_test =  sc.transform(X_test)

model_name = "sklearn-knn-classification"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/production")
y_pred= model.predict(X_test)

score = f1_score(y_pred=y_pred, y_true=y_test, average="weighted")
print("Score of loaded model:", score)