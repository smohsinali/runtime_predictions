import os
import re
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pymc3 import Model
from tabulate import tabulate
from mcmc import mcmc_predict, mcmc_fit
from data import load_data_pred, load_training_data
from hammer import hammer_fit
import pdb


def plot(x_features, x, y, y_predicted, data_pred, data_train):
    figtxt = "\n# features:" + str(int(x_features[0]))
    plt.plot(x, y, label="True-Runtime")
    colors = ["yellow", "red", "green"]
    table_row = None
    row = []
    keys = []
    row.append(pd.DataFrame({data_pred: [x[-1], x_features[-1]]},
                                index=['N', 'K']))
    keys.append('Dimensons')
    for i, predicted in enumerate(y_predicted):
        y_pred = predicted[0]
        y_pred_upper = predicted[1]
        y_pred_lower = predicted[2]
        # plt.plot(x, y_cal, label=y[3])
        plt.fill_between(x, y_pred_upper, y_pred_lower, facecolor=colors[i], alpha=0.2, label=predicted[3])
        print("DS:%s real:%0.3f predicted:%0.3f pU:%0.3f pL:%0.3f" %
              (data_pred.rjust(13), y[-1], y_pred[-1], y_pred_upper[-1], y_pred_lower[-1]))
        row.append(pd.DataFrame({data_pred: [y[-1], y_pred[-1], 100*(abs(y[-1]-y_pred[-1]))/y[-1]]},
                                index=['true', 'pred.', '%diff']))
        keys.append(predicted[3])

    plt.xlabel("Number of Samples")
    plt.ylabel("Time(s)")
    plt.title("Dataset:" + data_pred + figtxt + "\nPrediction using " + data_train, fontsize=12)
    plt.legend(loc='best', fontsize=8)

    # plt.figtext(10, 10, figtxt)
    # plt.savefig("results/mc_" + data_name_pred + "_pred_highres_wo2.png", dpi=300)
    plt.savefig("results/mc_" + data_pred + "_pred_n.png", dpi=200)
    plt.close()
    table = pd.concat(row, keys=keys).T
    # pdb.set_trace()
    return table


if __name__ == "__main__":
    # equations = ["w0+w1K+w2(KNlogN)", "w0+w1K+w2(KN²logN)", "w0+w1K+w2(KN^w3logN)"]
    equations = ["w0 + w1*(K*N*(logN)^2)", "w0 + w1*K + w2*(K*N^(2)*logN)", "w0 + w1*K + w2*(K*N^(w3)*logN)"]
    used_eq = [1, 2]  # likelihood function has multiple versions of eqs. here define which ones will be used.

    # find and load training data files
    x_train, y_train, data_name_train = load_training_data()

    total_data_size = len(y_train)
    print("total_data_size:", total_data_size)

    y_predicted = list()

    test_folder_path = "/home/alis/Desktop/project/runtime_prediction/runtimes/test/sgd"
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
        table = pd.concat([table, plot(x1_features, x2_size, y_runtime, y_predicted, data_name_pred, data_name_train)])
        y_predicted = list()

    table.to_html("table2.html")
    print("end")
    # pdb.set_trace()




