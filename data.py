import os
import re
import sys
import random
import numpy as np
from sklearn.cross_validation import train_test_split


def load_training_data():
    name = next(os.walk(os.path.abspath("runtimes/")))[2]
    names = sorted(list(set([x[16:-3] for x in name if re.search('np', x)])))
    sets = dict()
    for i, j in enumerate(names):
        print(i, ":", j)
        sets[i] = j
    print("Select Dataset which will be used for predicting runtimes.")
    # ids = input("Enter data id:")
    ids = "2"
    if re.search("[A-Za-z]", ids):
        print("Error: Please enter id in numbers only")
        sys.exit()
    ids = int(ids)
    if ids < 0 or ids >= len(names):
        print("Wrong id: no data set with this id exist")
        sys.exit()

    files = [x for x in name if re.search(sets[ids] + ".np", x)]

    if len(files) < 2:
        print("apparently one of files is missing, please investigate")
        sys.exit()

    x_path = os.path.join("runtimes/", list(filter(lambda x: x.startswith("x"), files))[0])
    y_path = os.path.join("runtimes/", list(filter(lambda x: x.startswith("y"), files))[0])
    print("Loading following data files for dataset " + sets[ids] + ":\n", x_path, y_path)
    x_runtime = np.loadtxt(x_path)
    y_runtime = np.loadtxt(y_path)
    y_runtime = np.around(y_runtime, decimals=4)

    return x_runtime, y_runtime, sets[ids]


def load_data_pred(data_name_pred, test_folder_path):
    # for testing only (refactor later)
    # print("Dataset Name which will be used for plotting reference time:", data_name_pred)
    x_runtime = np.loadtxt(os.path.join(test_folder_path, "x_runtime_train_" + data_name_pred + ".np"))
    y_runtime = np.loadtxt(os.path.join(test_folder_path, "y_runtime_train_" + data_name_pred + ".np"))
    y_runtime = np.around(y_runtime, decimals=4)
    x1_features = x_runtime[:, 1]
    x2_size = x_runtime[:, 0]

    return x1_features, x2_size, y_runtime


def process_train_data(x_data, y_data, a, dtype=0):
    ratio = a / 100.0
    ratio = 1
    print("ratio:", ratio)
    if dtype == 0:
        x_train = x_data[0:a]
        y_train = y_data[0:a]

    elif dtype == 1:
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                            train_size=ratio, random_state=random.randint(1, 100))

    return x_train, y_train