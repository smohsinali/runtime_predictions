'''
This file provides helper functions to load data sets in different scenarios.
'''
import os
import re
import sys
import random
import numpy as np
from sklearn.model_selection import train_test_split


def datasets_names(folder_path="runtimes/train/sgd_graphs"):
    '''
    Finds all datasets in a given folder and return list of their names
    :param folder_path: path of folder where to look for datasets
    :return: folder_path and list of datasets
    '''
    name = next(os.walk(os.path.abspath(folder_path)))[2]
    dataset_names = sorted(list(set([x[16:-3] for x in name if re.search('np', x)])))
    return folder_path, dataset_names


def load_dataset(dataset_name, folder_path="runtimes/train"):
    '''
    Loads given dataset in np arrays and returns it
    :param dataset_name: name of dataset
    :param folder_path: folder path of dataset
    :return: np arrays of data
    '''
    x_path = os.path.join(folder_path, "x_runtime_train_" + dataset_name + ".np")
    y_path = os.path.join(folder_path, "y_runtime_train_" + dataset_name + ".np")
    print("Loading following data files for dataset ", x_path, y_path)
    x_runtime = np.loadtxt(x_path)
    y_runtime = np.loadtxt(y_path)
    y_runtime = np.around(y_runtime, decimals=4)

    return x_runtime, y_runtime


def load_training_data(folder_path="runtimes"):
    '''
    Searches for all datasets in given directory and loads the one which user selects.
    This function is used for selecting and laoding the dataset which is used for
    :param folder_path: path of the folder where it searches for datasets in .np format
    :return: x_runtime_features, y_runtime, name_of_dataset
    '''
    name = next(os.walk(os.path.abspath(folder_path)))[2]
    names = sorted(list(set([x[16:-3] for x in name if re.search('np', x)])))
    sets = dict()
    for i, j in enumerate(names):
        print(i, ":", j)
        sets[i] = j
    print("Select Dataset which will be used for predicting runtimes.")
    ids = input("Enter data id:")

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
    x_runtime_features = np.loadtxt(x_path)
    y_runtime = np.loadtxt(y_path)
    y_runtime = np.around(y_runtime, decimals=8)

    return x_runtime_features, y_runtime, sets[ids]


def load_test_data(data_name_pred, test_folder_path):
    '''
    This function is used for individually loading dataset for testing the learned model
    :param data_name_pred: name of dataset that needs to be loaded
    :param test_folder_path: path of folder where the dataset is
    :return: dataset in np arrays
    '''
    # for testing only (refactor later)
    # print("Dataset Name which will be used for plotting reference time:", data_name_pred)
    x_runtime = np.loadtxt(os.path.join(test_folder_path, "x_runtime_train_" + data_name_pred + ".np"))
    y_runtime = np.loadtxt(os.path.join(test_folder_path, "y_runtime_train_" + data_name_pred + ".np"))
    y_runtime = np.around(y_runtime, decimals=8)
    x1_features = x_runtime[:, 1]
    x2_size = x_runtime[:, 0]

    return x1_features, x2_size, y_runtime


def process_train_data(x_data, y_data, percent_data, dtype=0):
    '''
    Divides data in training and test sets
    :param x_data: x_data
    :param y_data: y_data
    :param percent_data: percent of data needed
    :param dtype: 0 if samples are needed from beginning, 1 if random samples(not used anywhere)
    :return:
    '''

    x_train, x_test, y_train, y_test = [[],[],[],[]]
    a = int(percent_data * len(y_data))

    print("ratio:", a)
    if dtype == 0:
        # x_test and y_test start from 0 or a
        x_train = x_data[0:a]
        # x_test = x_data[a:]
        x_test = x_data[:]
        y_train = y_data[0:a]
        # y_test = y_data[a:]
        y_test = y_data[:]

    elif dtype == 1:
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                            train_size=percent_data, random_state=random.randint(1, 100))

    return x_train, x_test, y_train, y_test


# def plot_predictions(x_features, x, y, y_predicted, data_pred, data_train):
#     figtxt = "\n# features:" + str(int(x_features[0]))
#     plt.plot(x, y, label="True-Runtime")
#     colors = ["yellow", "red", "green"]
#     table_row = None
#     row = []
#     keys = []
#     for i, predicted in enumerate(y_predicted):
#         y_pred = predicted[0]
#         y_pred_upper = predicted[1]
#         y_pred_lower = predicted[2]
#         # plt.plot(x, y_cal, label=y[3])
#         plt.fill_between(x, y_pred_upper, y_pred_lower, facecolor=colors[i], alpha=0.2, label=predicted[3])
#         print("DS:%s real:%0.3f predicted:%0.3f pU:%0.3f pL:%0.3f" %
#               (data_pred.rjust(13), y[-1], y_pred[-1], y_pred_upper[-1], y_pred_lower[-1]))
#         row.append(pd.DataFrame({data_pred: [y[-1], y_pred[-1], 100*(abs(y[-1]-y_pred[-1]))/y[-1]]},
#                                 index=['true', 'pred.', '%diff']))
#         keys.append(predicted[3])
#
#     plt.xlabel("Number of Samples")
#     plt.ylabel("Time(s)")
#     plt.title("Dataset:" + data_pred + figtxt + "\nPrediction using " + data_train, fontsize=12)
#     plt.legend(loc='best', fontsize=8)
#
#     # plt.figtext(10, 10, figtxt)
#     # plt.savefig("results/mc_" + data_name_pred + "_pred_highres_wo2.png", dpi=300)
#     plt.savefig("results/mc_" + data_pred + "_pred_n.png", dpi=200)
#     plt.close()
#     table = pd.concat(row, keys=keys).T
#     # pdb.set_trace()
#     return table