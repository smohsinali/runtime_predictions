import numpy
import random
from numpy import arange
from sklearn import metrics
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn import tree
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
import sklearn.preprocessing as preprocessing
import sys
import pandas as pd
import numpy as np
import sklearn as skl
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
import sklearn.cross_validation as cross_validation
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


def ImportSparse(datasetname, settype='train', binary_formatted_input_file=False, numcols=[]):
        ds = []
        load_start_time = time.time()
        print("filename:%s" % datasetname)
        with open(datasetname, 'r') as readfile:
            for line in readfile:
                print("line:%s" % line)
                if line == 1 or line == -1:
                    item_to_append = line
                else:
                    item_to_append = line.split(' ')[:-2]
                # print(len(item_to_append))
                ds.append(item_to_append)
        i = 0
        rows, cols, vals = [], [], []
        for d in ds:
            # print("d:%s" %d)
            for loc in d:
                print("loc:%s" %loc)
                rows.append(int(i))
                if binary_formatted_input_file:
                    cols.append(int(loc))
                    vals.append(int(1))
                else:
                    entry = loc.split(':')
                    cols.append(int(entry[0]))
                    vals.append(int(entry[1]))
            i += 1
        if numcols == []: numcols = max(cols) + 1
        # print('\n' + 'Used {0:.3f} '.format((time.time() - load_start_time)) + \
        #       'seconds to load one part of the ' + datasetname + \
        #       ' data into a sparse matrix format prior to inflating.')
        print("shape  of dexter %s data:%s,%s" %(settype, i, numcols))
        return csc_matrix((vals, (rows, cols)), shape=(i, numcols))


def sparsedataload(data_set_name, train_test_validate_designation, data_file_extension, native_binary=True):
        print("datatype:%s" % data_file_extension)
        data_file_name = "datasets/mldata/" + data_set_name + '_' + train_test_validate_designation + '.' + data_file_extension
        if native_binary:
            loaded_sparse_data = ImportSparse(data_file_name,
                                              settype=train_test_validate_designation,
                                              binary_formatted_input_file=True)
        else:
            loaded_sparse_data = ImportSparse(data_file_name, settype=train_test_validate_designation)
        return csc_matrix.todense(loaded_sparse_data)


def load_data(dataset, data_type="train"):

    if dataset == "dexter":
        print("Dataset %s selected" % dataset)
        data_set_name_to_use = 'dexter'
        data_set_name_to_save = 'Dexter'
        train_data = sparsedataload(data_set_name_to_use, data_type, 'data', native_binary=False)
        train_label = sparsedataload(data_set_name_to_use, data_type, 'labels', native_binary=True)
        print("Finished Loading Data")

    elif dataset == "mnist":
        print("Dataset %s selected" % dataset)
        mnist = fetch_mldata('MNIST original')
        train_data = mnist.data
        train_label = mnist.target
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
        print("Finished Loading Data")
        # print(train_data[0:10])
        # sys.exit()

    elif dataset == "covertype":
        print("Dataset %s selected" % dataset)
        np.set_printoptions(formatter={"float": lambda x: "%0.0f" % x})
        data = np.loadtxt("datasets/mldata/covtype.data", delimiter=",")
        print(data[:, -1])
        train_data = data[:, :-1]
        train_label = data[:, -1]
        print("Finished Loading Data")

    else:
        print("Dataset %s does not exist, now existing" % dataset)
        sys.exit()
    print(train_data[0:1])
    print("Dimensions of data are %s,%s" %(train_data.shape[0], train_data.shape[1]))
    # print("Dimensions of data are %s,%s" %(len(train_data), len((train_data[0]))))
    # sys.exit()
    return train_data, train_label


def run():
    print("start")

    # Define training and testing sets
    random.seed(0)

    training_size = list()
    training_time = list()
    dataset = "mnist"
    train_data, train_label = load_data(dataset)
    # print(train_data[0:10])

    for ts in range(5, 95, 5):
        tr_s = ts/100.0
        x_train, x_test, y_train, y_test = train_test_split(train_data,
                                                            train_label,
                                                            train_size=tr_s, random_state=42)
        print("train_ratio:", tr_s, " length of training data:", len(x_train))
        clf = tree.DecisionTreeClassifier()

        start_time = time.time()
        clf = clf.fit(x_train, y_train)
        end_time = time.time()
        training_size.append(tr_s)
        training_time.append(end_time - start_time)
        print("time taken:", end_time - start_time)

    plt.xlabel("% training data")
    plt.ylabel("Time/s")
    plt.title("Time taken to train data")
    plt.plot(training_size, training_time)
    plt.savefig('DT time ' + dataset + '.png')

    print("end")


def single_run():
    print("start")

    # Define training and testing sets
    random.seed(0)

    training_size = list()
    training_time = list()
    dataset = "mnist"
    train_data, train_label = load_data(dataset, "train")

    x_train, x_test, y_train, y_test = train_test_split(train_data,
                                                            train_label,
                                                            train_size=.9, random_state=42)

    print("length of training data:", len(x_train))
    clf = tree.DecisionTreeClassifier()

    start_time = time.time()
    clf = clf.fit(x_train, y_train)
    end_time = time.time()

    training_time.append(end_time - start_time)
    x_valid, y_valid = load_data(dataset, "valid")

    y_predicted = clf.predict(x_valid)
    accuracy = metrics.accuracy_score(y_predicted, y_valid)
    print("time taken:", end_time - start_time)
    print("Accuracy:%s" % accuracy)
    print("end")

if __name__ == "__main__":
    print("Project ORACLE\nSyed Mohsin Ali\n")
    single_run()
    # run()


