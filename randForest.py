import numpy as np
import random
from numpy import arange
from sklearn import metrics
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
import time


def run():
    mnist = fetch_mldata('MNIST original')

    # Trunk the data
    n_train = 60000
    n_test = 10000

    # Define training and testing sets
    indices = arange(len(mnist.data))
    random.seed(0)
    train_idx = arange(0, n_train)
    test_idx = arange(n_train+1, n_train+n_test)

    x_train, y_train = mnist.data[train_idx], mnist.target[train_idx]
    x_test, y_test = mnist.data[test_idx], mnist.target[test_idx]
    # Apply a learning algorithm
    print("Applying a learning algorithm...")
    clf = RandomForestClassifier(n_estimators=10,n_jobs=2)
    clf.fit(x_train, y_train)

    # Make a prediction
    print("Making predictions...")
    y_pred = clf.predict(x_test)

    # Evaluate the prediction
    print("Evaluating results...")
    print("Accuracy: \t", metrics.accuracy_score(y_test, y_pred))
    # print("Recall: \t", metrics.recall_score(y_test, y_pred))
    # print("F1 score: \t", metrics.f1_score(y_test, y_pred))
    # print("Mean accuracy: \t", clf.score(X_test, y_test))


if __name__ == "__main__":
    print("Project ORACLE\nSyed Mohsin Ali\n")
    np.set_printoptions(formatter={"float": lambda x: "%0.4f" % x})
    start_time = time.time()
    results = run()
    end_time = time.time()
    print("Overall running time:", end_time - start_time)