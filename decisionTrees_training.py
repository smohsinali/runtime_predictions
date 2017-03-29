# Mohsin Ali
# This scripts is used to train decision trees on  mnist, covertype, dexter and gisette datasets
# The training time of decision trees is recorded for different data sizes of these datasets and saved in numpy arrays

import random
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import time
import timeit
import sys
import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
import pickle
import sklearn.metrics as metrics
import sklearn.tree as tree


# Encode the categorical features as numbers
def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        # print("column:", column)
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            # print("result columns \n%s end" % result[column])
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders


def load_data(dataset, data_type="train"):
    print()
    if dataset == "dexter":
        print("Dataset %s selected" % dataset)
        data = open("datasets/mldata/dexter_" + data_type + ".data", "r")
        labels = open("datasets/mldata/dexter_" + data_type + ".labels", "r")

        lines = data.readlines()

        values = np.array([0])
        rows = np.array([0])
        cols = np.array([0])

        for i in range(len(lines)):
            line_list = lines[i].split()
            for j in line_list:
                vals = j.split(":")
                rows = np.insert(rows, 0, int(i))
                cols = np.insert(cols, 0, int(vals[0]))
                values = np.insert(values, 0, int(vals[1]))

        train_data = csr_matrix((values, (rows, cols)))
        train_label = np.loadtxt(labels)

        shape = train_data.toarray().shape
        print("Finished Loading Data")

    elif dataset == "mnist":
        print("Dataset %s selected" % dataset)
        mnist = fetch_mldata('MNIST original')
        train_data = mnist.data
        train_label = mnist.target
        shape = train_data.shape
        print(shape)
        print("Finished Loading Data")

    elif dataset == "adult":
        print("Dataset %s selected" % dataset)
        original_data = pd.read_csv(
            "datasets/mldata/adult (copy).data",
            names=[
                "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
                "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                "Hours per week", "Country", "Target"],
            sep=r'\s*,\s*',
            engine='python',
            na_values="?")
        # Calculate the correlation and plot it
        encoded_data, encoders = number_encode_features(original_data)
        train_data = encoded_data[encoded_data.columns.difference(["Target"])]
        train_label = encoded_data["Target"]
        shape = train_data.shape
        print("Finished Loading Data")
        # print(train_data[0:10])
        # sys.exit()

    elif dataset == "covertype":
        print("Dataset %s selected" % dataset)
        # np.set_printoptions(formatter={"float": lambda x: "%0.0f" % x})
        data = np.loadtxt("datasets/mldata/covtype.data", delimiter=",")
        print(data[:, -1])
        train_data = data[:, :-1]
        train_label = data[:, -1]
        shape = train_data.shape
        print("Finished Loading Data")

    elif dataset == "gisette":
        print("Dataset %s selected" % dataset)
        # data = np.loadtxt("datasets/mldata/gisette_train.data")
        labels = np.loadtxt("datasets/mldata/gisette_train.labels")

        # data = csr_matrix(data)
        # with open("datasets/mldata/gissete_pickle.data", "wb") as outfile:
        #     pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)

        with open("datasets/mldata/gisette_pickle.data", "rb") as infile:
            train_data = pickle.load(infile)
        train_label = labels
        shape = train_data.toarray().shape
        print("Finished loading dataset")

    else:
        print("Dataset %s does not exist, now existing" % dataset)
        sys.exit()

    # print(train_data[0])
    print("Dimensions of data are %s,%s" % (shape[0], shape[1]))
    # sys.exit()
    return train_data, train_label, shape


def run(data_sets, data_name):

    num_runs = input("Enter number of runs:")
    num_runs = int(num_runs)

    # datasize in percentage
    data_size = 1
    total_data_size = data_size * num_runs * len(data_sets)
    print("total data size:%d" % total_data_size)
    x_runtime_train = np.zeros((total_data_size, 4))
    y_runtime_train = np.zeros((total_data_size, 1))

    for d_num, data in enumerate(data_sets):
        train_data, train_label, shape = load_data(data)
        for i in range(num_runs):
            for ts in range(1, data_size + 1, 1):
                tr_s = ts / 100.0
                x_train, x_test, y_train, y_test = train_test_split(train_data,
                                                                    train_label,
                                                                    train_size=tr_s, random_state=random.randint(1, 100))
                # print("train_ratio:", tr_s, " length of training data:", len(x_train))

                clf = tree.DecisionTreeClassifier()

                start_time = time.process_time()
                # time_taken = timeit.timeit(clf.fit(x_train, y_train))
                clf = clf.fit(x_train, y_train)
                end_time = time.process_time()
                time_taken = end_time - start_time

                # x_runtime_train[ts - 1] = [x_train.shape[0], x_train.shape[1]]
                # y_runtime_train[ts - 1] += time_taken
                index = num_runs * data_size * d_num + (data_size * i + (ts - 1))
                print("index", index, "data:", data, " run:", i, " %data:", tr_s, " time taken:", time_taken,
                      " datasize:", x_train.shape[0], x_train.shape[1],
                      " depth:", clf.tree_.max_depth, " # of nodes:", clf.tree_.node_count)
                x_runtime_train[index] = [x_train.shape[0], x_train.shape[1], clf.tree_.max_depth, clf.tree_.node_count]
                y_runtime_train[index] = time_taken

    # take average of runtimes
    y_runtime_train /= num_runs

    plt.xlabel("% training data")
    plt.ylabel("Time/s")
    plt.title("Time taken to train data")
    # plt.plot(np.array([x for x in range(1, total_data_size + 1, 1)]) / 100.0, y_runtime_train)
    plt.savefig('results/DT time ' + data_name + ' ' + str(num_runs) + ' runs.png')

    np.savetxt("runtimes/x_runtime_train_" + data_name + ".np", x_runtime_train, fmt="%0.1f")
    np.savetxt("runtimes/y_runtime_train_" + data_name + ".np", y_runtime_train, fmt="%0.5f")
    # np.savetxt("runtimes/y_runtime_train_" + dataset + ".np", y_runtime_train, fmt="%0.5f")

    print("end")

if __name__ == "__main__":
    print("Project ORACLE\nSyed Mohsin Ali\n")
    # single_run()
    np.set_printoptions(formatter={"float": lambda x: "%0.4f" % x})
    print("Please select Dataset by entering its numeric id:")
    print("1:Mnist\n"
          "2:Covertype\n"
          "3:Adult\n"
          "4:Gissete\n"
          "5:Mnist, Covertype and Adult\n"
          "6:Mnist, Covertype and Gisette\n"
          "7:Mnist, Adult and Gisette\n"
          "8:Covertype, Adult and Gisette\n"
          "9:Covertype, Adult, Gisette and Mnist\n"

          )
    ids = input("Enter Dataset id:")
    ids = int(ids)
    if ids == 1:
        datasets = ["mnist"]
        dataname = "mnist"
    elif ids == 2:
        datasets = ["covertype"]
        dataname = "covertype"
    elif ids == 3:
        datasets = ["adult"]
        dataname = "adult"
    elif ids == 4:
        datasets = ["gisette"]
        dataname = "gisette"
    elif ids == 5:
        datasets = ["mnist", "covertype", "adult"]
        dataname = "mca"
    elif ids == 6:
        datasets = ["mnist", "covertype", "gisette"]
        dataname = "mcg"
    elif ids == 7:
        datasets = ["mnist", "adult", "gisette"]
        dataname = "mag"
    elif ids == 8:
        datasets = ["covertype", "adult", "gisette"]
        dataname = "cag"
    elif ids == 9:
        datasets = ["covertype", "adult", "gisette", "mnist"]
        dataname = "cagm"
    else:
        print("\nwrong id selected,exiting program")
        sys.exit()

    run(datasets, dataname)
    # run()
