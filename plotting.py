import matplotlib.pyplot as plt
import pandas as pd

def plot_prediction(x_features, x_datasize, y_real, y_predicted, data_pred, data_train):
    '''

    :param x_features: array of # of features (all entries are same)
    :param x_datasize: array that contains different number of samples used for training
    :param y_real: array of experimental runtimes
    :param y_predicted: array of predicted lower, upper and avg runtimes
    :param data_pred: name of dataset on which predictions are made
    :param data_train: name of dataset used for training
    :return: a row of following table

     %Data     || Dimensions  || Equations
               || N|K         || True | Pred | %diff
     DS name   ||             ||

    '''
    figtxt = "\n# features:" + str(int(x_features[0]))
    plt.plot(x_datasize, y_real, label="True-Runtime")
    colors = ["yellow", "red", "green"]
    table_row = None
    row = []
    keys = []
    row.append(pd.DataFrame({data_pred: [x_datasize[-1], x_features[-1]]},
                            index=['N', 'K']))
    keys.append('Dimensons')
    for i, predicted in enumerate(y_predicted):
        y_pred = predicted[0]
        y_pred_upper = predicted[1]
        y_pred_lower = predicted[2]
        # plt.plot(x, y_cal, label=y[3])
        plt.fill_between(x_datasize, y_pred_upper, y_pred_lower, facecolor=colors[i], alpha=0.2, label=predicted[3])
        print("DS:%s real:%0.3f predicted:%0.3f pU:%0.3f pL:%0.3f" %
              (data_pred.rjust(13), y_real[-1], y_pred[-1], y_pred_upper[-1], y_pred_lower[-1]))
        row.append(pd.DataFrame({data_pred: [y_real[-1], y_pred[-1], 100 * (abs(y_real[-1] - y_pred[-1])) / y_real[-1]]},
                                index=['true', 'pred.', '%diff']))
        keys.append(predicted[3])

    plt.xlabel("Number of Samples")
    plt.ylabel("Time(s)")
    plt.title("Dataset:" + data_pred + figtxt + "\nPrediction using " + data_train, fontsize=12)
    plt.legend(loc='best', fontsize=8)

    plt.savefig("results/mc_" + data_pred + "_pred_n.png", dpi=200)
    plt.close()
    table = pd.concat(row, keys=keys).T
    # pdb.set_trace()
    return table


def plot_get_hp_table(x_features, x_datasize, y_real, y_predicted, hp_values, data_pred, data_train):
    '''

    :param x_features: array of # of features (all entries are same)
    :param x_datasize: array that contains different number of samples used for training
    :param y_real: array of experimental runtimes
    :param y_predicted: array of predicted lower, upper and avg runtimes
    :param hp_values: array of mean values of hyperparameters
    :param data_pred: name of dataset on which predictions are made
    :param data_train: name of dataset used for training
    :return: a row of the following table

     %Data     || Dimensions  || Equations
               || N|K         || param1 | param2 | %diff
     DS name   ||             ||

    '''
    figtxt = "\n# features:" + str(int(x_features[0]))
    plt.plot(x_datasize, y_real, label="True-Runtime")
    colors = ["yellow", "red", "green"]
    table_row = None
    row = []
    keys = []
    row.append(pd.DataFrame({data_pred: [x_datasize[-1], x_features[-1]]},
                            index=['N', 'K']))
    keys.append('Dimensons')
    for i, predicted in enumerate(y_predicted):
        y_pred = predicted[0]
        y_pred_upper = predicted[1]
        y_pred_lower = predicted[2]
        # plt.plot(x, y_cal, label=y[3])
        plt.fill_between(x_datasize, y_pred_upper, y_pred_lower, facecolor=colors[i], alpha=0.2, label=predicted[3])
        print("DS:%s real:%0.3f predicted:%0.3f pU:%0.3f pL:%0.3f" %
              (data_pred.rjust(13), y_real[-1], y_pred[-1], y_pred_upper[-1], y_pred_lower[-1]))
        row.append(pd.DataFrame({data_pred: [hp_values[0], hp_values[1], 100 * (abs(y_real[-1] - y_pred[-1])) / y_real[-1]]},
                                index=['alpha', 'beta', '%diff']))
        keys.append(predicted[3] + '(19%)')

    plt.xlabel("Number of Samples")
    plt.ylabel("Time(s)")
    plt.title("Dataset:" + data_pred + figtxt + "\nPrediction using " + data_train, fontsize=12)
    plt.legend(loc='best', fontsize=8)

    plt.savefig("results/mc_" + data_pred + "_pred_n.png", dpi=200)
    plt.close()
    table = pd.concat(row, keys=keys).T
    # pdb.set_trace()
    return table
