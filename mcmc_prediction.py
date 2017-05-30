import os
import re
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pymc3 import Model
from mcmc import mcmc_predict, mcmc_fit
from data import load_test_data, load_training_data
from plotting import plot_prediction, plot_get_hp_table

"""
This program trains a single model and then uses it to predict on some test datasets.
In the report we show that this idea does not work well.
"""
if __name__ == "__main__":

    equations = {
        'dt_lower': 'a + bN(log(N))^2',
        'dt_avg': 'a + bKN(logN)^2',
        'dt_upper': 'a + bKN^2(log(N))',
        'rf_lower': 'a + bN*sqrt(K)*(log(N))^2',
        'rf_lower_2': 'a + bN(log(N))^2 + c(sqrt(K))',
        'rf_avg': 'a + bKN(log(N))^2',
        'rf_upper': 'a + bN(log(N))^2',
        'rf_upper_2': 'a + bN^c(log(N))^2',
        'sgd_lower': 'a + bKN',
        'sgd_lower_2': 'a + bN^1.3',
        'sgd_avg': 'a + bN + cK',
        'sgd_upper': 'a + bN^1.2 + cK'

    }

    # likelihood function has multiple versions of eqs. here define which ones will be used.
    used_eq = [
        # 'dt_lower',
        'dt_avg',
        'dt_upper',
        # 'rf_lower',
        # 'rf_lower_2',
        # 'rf_avg',
        # 'rf_upper',
        # 'rf_upper_2',
        # 'sgd_lower',
        # 'sgd_lower_2',
        # 'sgd_avg',
        # 'sgd_upper'
    ]

    train_folder_path = 'runtimes'
    # train_folder_path = 'runtimes/train/rf_graphs'
    # train_folder_path = 'runtimes/train/dt_graphs'
    # train_folder_path = 'runtimes/train/sgd_graphs'

    # train_folder_path = 'runtimes/all_sgd/0/train'
    # test_folder_path = 'runtimes/all_sgd/0/test'

    # find and load training data files
    x_train, y_train, dataset_train = load_training_data(folder_path=train_folder_path)

    total_data_size = len(y_train)
    print("total_data_size:", total_data_size)

    y_predicted = list()

    test_folder_path = "runtimes/test/dt_graphs"
    # test_folder_path = "runtime_prediction/runtimes/test/dt_graphs"
    # test_folder_path = "runtime_prediction/runtimes/test/rf_graphs"
    files = next(os.walk(os.path.abspath(test_folder_path)))[2]
    test_data_sets = sorted(list(set([x[16:-3] for x in files if re.search('np', x)])))

    # x_train, y_train = process_train_data(x_data, y_data, 100.0, 1)
    # x_train = x_data
    # y_train = y_data
    # fit the model
    trace_pymc = mcmc_fit(x_train, y_train, used_eq, dataset_train)
    # trace_hammer = hammer_fit(x_train, y_train, used_eq)
    table = pd.DataFrame()

    for dataset_test in test_data_sets:
        # load dataset on which prediction of runtimes is desired
        x1_features, x2_size, y_runtime = load_test_data(dataset_test, test_folder_path)

        for equation in used_eq:
            # load the model from pickle file
            model = Model()
            with model:
                trace = pickle.load(open("results/model_" + str(equation) + ".pickle", "rb"))
            # predict on data_name_pred
            y_pred, y_pred_upper, y_pred_lower = mcmc_predict(trace, x1_features, x2_size, equation)
            y_predicted.append([y_pred, y_pred_upper, y_pred_lower, equations[equation]])

        # plot data
        table = pd.concat([table, plot_prediction(x1_features, x2_size, y_runtime, y_predicted, dataset_test, dataset_train)])
        y_predicted = list()
    # table.to_html("table.csv")
    table.to_html("dt_test.html")
    print("end")
    # pdb.set_trace()




