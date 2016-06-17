import os
import re
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pymc3 import Model

from mcmc import mcmc_predict, mcmc_fit
from data import load_data_pred, load_training_data
from hammer import hammer_fit


def plot(x_features, x, y, y_predicted, data_pred, data_train):
    figtxt = "\n# features:" + str(int(x_features[0]))
    plt.plot(x, y, label="True-Runtime")
    colors = ["yellow", "red", "green"]
    for i, predicted in enumerate(y_predicted):
        y_pred = predicted[0]
        y_pred_upper = predicted[1]
        y_pred_lower = predicted[2]
        # plt.plot(x, y_cal, label=y[3])
        plt.fill_between(x, y_pred_upper, y_pred_lower, facecolor=colors[i], alpha=0.2, label=predicted[3])
        print("DS:%s real:%0.3f predicted:%0.3f pU:%0.3f pL:%0.3f" %
              (data_pred.rjust(13), y[-1], y_pred[-1], y_pred_upper[-1], y_pred_lower[-1]))

    plt.xlabel("Number of Samples")
    plt.ylabel("Time(s)")
    plt.title("Dataset:" + data_pred + figtxt + "\nPrediction using " + data_train, fontsize=12)
    plt.legend(loc='best', fontsize=8)

    # plt.figtext(10, 10, figtxt)
    # plt.savefig("results/mc_" + data_name_pred + "_pred_highres_wo2.png", dpi=300)
    plt.savefig("results/mc_" + data_pred + "_pred_n.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    # equations = ["w0+w1K+w2(KNlogN)", "w0+w1K+w2(KN²logN)", "w0+w1K+w2(KN^w3logN)"]
    equations = ["w0+w[1](KNlog²N)", "w0+w1K+w2(KN²logN)", "w0+w1K+w2(KN^w3logN)"]
    used_eq = [1, 2, 3]  # likelihood function has multiple versions of eqs. here define which ones will be used.

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

    # # save the learned mode in pickle file
    # pickle.dump(trace_tmp, open("results/model.pickle", "wb"), protocol=-1)
    #
    # # move plot about learned param values to results folder
    # os.rename("mcmc.png", "results/mcmc_" + data_name_train + "N.png")

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
        plot(x1_features, x2_size, y_runtime, y_predicted, data_name_pred, data_name_train)
        y_predicted = list()
