import matplotlib.pyplot as plt
import pandas as pd

def plot_prediction(x_features, x_datasize, y_true, y_predicted, data_test, data_train):
    '''

    :param x_features: array of # of features (all entries are same)
    :param x_datasize: array that contains different number of samples used for training
    :param y_true: array of experimental runtimes
    :param y_predicted: array of predicted lower, upper and avg runtimes
    :param data_test: name of dataset on which predictions are made
    :param data_train: name of dataset used for training
    :return: a row of following table

     %Data     || Dimensions  || Equations
               || N|K         || True | Pred | %diff | %uncertain
     DS name   ||             ||

    '''
    figtxt = "\n# of Features:" + str(int(x_features[0]))
    plt.plot(x_datasize, y_true, label="True-Runtime")
    colors = ["yellow", "red", "green", "blue", "orange", "purple"]
    table_row = None
    row = []
    keys = []
    row.append(pd.DataFrame({data_test: [x_datasize[-1], x_features[-1]]},
                            index=['N', 'K']))
    keys.append('Dimensons')
    for i, predicted in enumerate(y_predicted):
        y_pred = predicted[0]
        y_pred_upper = predicted[1]
        y_pred_lower = predicted[2]
        # plt.plot(x, y_cal, label=y[3])
        plt.fill_between(x_datasize, y_pred_upper, y_pred_lower, facecolor=colors[i], alpha=0.2, label=predicted[3])
        print("DS:%s real:%0.3f predicted:%0.3f pU:%0.3f pL:%0.3f" %
              (data_test.rjust(13), y_true[-1], y_pred[-1], y_pred_upper[-1], y_pred_lower[-1]))

        row.append(pd.DataFrame({data_test: [y_true[-1],
                                             y_pred[-1],
                                             100 * ((y_pred[-1] - y_true[-1])) / y_true[-1],
                                             100 * (abs(y_pred[-1] - y_pred_upper[-1])) / y_pred[-1]]
                                 },
                                index=['true', 'pred.', '%diff', '%uncertainity']))
        keys.append(predicted[3])

    plt.xlabel("Number of Samples")
    plt.ylabel("Time(s)")
    plt.title("Dataset:" + data_test + figtxt + "\nPrediction using " + data_train, fontsize=12)
    plt.legend(loc='best', fontsize=8)

    plt.savefig("results/mc_" + data_test + "_predicted.png", dpi=200)
    plt.close()
    table = pd.concat(row, keys=keys).T
    # pdb.set_trace()
    return table


def plot_get_hp_table(x_features, x_datasize, y_true, y_predicted, hp_values, data_test, data_train, data_used):
    '''

    :param data_used: percentage data being usd for training the model
    :param x_features: array of # of features (all entries are same)
    :param x_datasize: array that contains different number of samples used for training
    :param y_true: array of experimental runtimes
    :param y_predicted: array of predicted lower, upper and avg runtimes
    :param hp_values: array of mean values of hyperparameters
    :param data_test: name of dataset on which predictions are made
    :param data_train: name of dataset used for training
    :return: a row of the following table

     %Data     || Dimensions  || Equations
               || N|K         || param1 | param2 | %diff
     DS name   ||             ||

    '''
    figtxt = "\n# features:" + str(int(x_features[0]))
    plt.plot(x_datasize, y_true, label="True-Runtime")
    colors = ["yellow", "red", "green"]
    table_row = None
    row = []
    keys = []
    row.append(pd.DataFrame({data_test: [x_datasize[-1], x_features[-1]]},
                            index=['N', 'K']))
    keys.append('Dimensons')
    for i, predicted in enumerate(y_predicted):
        y_pred = predicted[0]
        y_pred_upper = predicted[1]
        y_pred_lower = predicted[2]
        # plt.plot(x, y_cal, label=y[3])
        plt.fill_between(x_datasize, y_pred_upper, y_pred_lower, facecolor=colors[i], alpha=0.2, label=predicted[3])
        print("DS:%s real:%0.3f predicted:%0.3f pU:%0.3f pL:%0.3f" %
              (data_test.rjust(13), y_true[-1], y_pred[-1], y_pred_upper[-1], y_pred_lower[-1]))
        row.append(pd.DataFrame({data_test: [hp_values[0], hp_values[1], 100 * (abs(y_true[-1] - y_pred[-1])) / y_true[-1]]},
                                index=['alpha', 'beta', '%diff']))
        keys.append(predicted[3] + str(data_used) + '%')

    plt.xlabel("Number of Samples")
    plt.ylabel("Time(s)")
    plt.title("Dataset:" + data_test + figtxt + "\nPrediction using " + data_train, fontsize=12)
    plt.legend(loc='best', fontsize=8)

    plt.savefig("results/mc_" + data_test + "_" + str(data_used) + "_pred_n.png", dpi=200)
    plt.close()
    table = pd.concat(row, keys=keys).T
    # pdb.set_trace()
    return table

def plot_rf_results(x_features, x_datasize, y_true, y_predicted, data_test):
    plt.plot(x_datasize, y_true, label="True-Runtime")
    plt.plot(x_datasize, y_predicted, label="Predicted-Runtime")

    row = []
    keys = []
    row.append(pd.DataFrame({data_test: [x_datasize[-1], x_features[-1]]},
                            index=['N', 'K']))
    keys.append('Dimensons')

    row.append(pd.DataFrame({data_test: [y_true[-1], y_predicted[-1], 100 * (abs(y_true[-1] - y_predicted[-1])) / y_true[-1]]},
                            index=['true', 'pred.', '%diff']))

    plt.xlabel("Number of Samples")
    plt.ylabel("Time(s)")
    plt.title("Dataset:" + data_test + "\nPrediction using All", fontsize=12)
    plt.legend(loc='best', fontsize=8)

    plt.savefig("results/mlp50kdt_" + data_test + "_pred.png", dpi=200)
    plt.close()
    table = pd.concat(row, keys=keys).T
    # pdb.set_trace()
    return table