from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from data import load_data_pred, load_training_data, data_sets_in_folder, load_data_set, process_train_data
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

    plt.plot(x, y_predicted, label="Predicted-Runtime")
    print("DS:%s real:%0.3f predicted:%0.3f" %
          (data_pred.rjust(13), y[-1], y_predicted[-1]))
    row.append(pd.DataFrame({data_pred: [y[-1], y_predicted[-1], 100*(abs(y[-1]-y_predicted[-1]))/y[-1]]},
                            index=['true', 'pred.', '%diff']))
    keys.append("RF")

    plt.xlabel("Number of Samples")
    plt.ylabel("Time(s)")
    plt.title("Dataset:" + data_pred + figtxt + "\nPrediction using " + data_train + "(RF)", fontsize=12)
    plt.legend(loc='best', fontsize=8)

    plt.savefig("results/mc_" + data_pred + "_pred_n_RF.png", dpi=200)
    plt.close()
    table = pd.concat(row, keys=keys).T
    # pdb.set_trace()
    return table


if __name__ == "__main__":
    table = pd.DataFrame()
    train_folder_path, training_data_sets = data_sets_in_folder()
    # find and load training data files
    for data_set in training_data_sets:
        print("dataset", data_set)
        x_data, y_data = load_data_set(data_set, train_folder_path)

        # sys.exit()
        total_data_size = len(y_data)
        print("total_data_size:", total_data_size)

        y_predicted = list()

        x_train, x_test, y_train, y_test = process_train_data(x_data, y_data, 0.28, 0)
        # fit the model
        rf_regr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                            max_depth=1, random_state=0, loss='ls').fit(x_train, y_train)

        # x1_features, x2_size, y_runtime = load_data_pred(data_name_pred, test_folder_path)
        # x1_features = x_test[:, 1]
        # x2_size = x_test[:, 0]
        # y_runtime = y_test

        x1_features = x_data[:, 1]
        x2_size = x_data[:, 0]
        y_runtime = y_data
        # predict on data_name_pred
        y_predicted = rf_regr.predict(x_data)

        table = pd.concat([table, plot(x1_features, x2_size, y_runtime, y_predicted, data_set, data_set)])
        y_predicted = list()
        print("end one ds")
    table.to_html("table_rf.html")
    print("end")
    # pdb.set_trace()





