import os
import re
import sys
import pickle
import numpy as np
import pandas as pd
from pymc3 import Model
from tabulate import tabulate
from mc2 import mcmc_fit, mcmc_predict
from data import load_data_pred, load_training_data, plot_predictions
from hammer import hammer_fit
import pdb


if __name__ == "__main__":
    # equations = ["w0+w1K+w2(KNlogN)", "w0+w1K+w2(KN²logN)", "w0+w1K+w2(KN^w3logN)"]
    equations = ["w0+w[1](KNlog²N)", "w0+w1K+w2(KN²logN)", "w0+w1K+w2(KN^w3logN)"]
    used_eq = [1]  # likelihood function has multiple versions of eqs. here define which ones will be used.

    # find and load training data files
    x_train, y_train, data_name_train = load_training_data()

    total_data_size = len(y_train)
    print("total_data_size:", total_data_size)

    y_predicted = list()

    test_folder_path = "/home/alis/Desktop/project/runtime_prediction/runtimes/test"
    files = next(os.walk(test_folder_path))[2]
    test_data_sets = sorted(list(set([x[16:-3] for x in files if re.search('np', x)])))

    # x_train, y_train = process_train_data(x_data, y_data, 100.0, 1)
    # x_train = x_data
    # y_train = y_data

    # fit the model
    trace_pymc = mcmc_fit(x_train, y_train, used_eq)
    # trace_hammer = hammer_fit(x_train, y_train, used_eq)
    table = pd.DataFrame()

    for data_name_pred in test_data_sets:
        # load dataset on which prediction of runtimes is desired
        x1_features, x2_size, y_runtime = load_data_pred(data_name_pred, test_folder_path)

        for equation in used_eq:
            # load the model from pickle file
            model = Model()
            with model:
                trace = pickle.load(open("results/model" + str(equation) + ".pickle", "rb"))

            # predict on data_name_pred
            y_pred, y_pred_upper, y_pred_lower = mcmc_predict(trace, x1_features, x2_size, equation)
            y_predicted.append([y_pred, y_pred_upper, y_pred_lower, equations[equation - 1]])

        # plot data
        table = pd.concat([table, plot_predictions(x1_features, x2_size, y_runtime, y_predicted, data_name_pred, data_name_train)])
        y_predicted = list()

    print("end")
    pdb.set_trace()




