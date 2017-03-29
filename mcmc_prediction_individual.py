import pickle
import pandas as pd
import numpy as np
from pymc3 import Model
from mcmc import mcmc_predict, mcmc_fit
from data import load_test_data, load_training_data, datasets_names, load_dataset, process_train_data
from plotting import plot_prediction, plot_get_hp_table

if __name__ == "__main__":
    # old_equations = ["w0+w1K+w2(KNlogN)", "w0+w1K+w2(KN²logN)", "w0+w1K+w2(KN^w3logN)"]
    # equations = ["w0 + w1*(K*N*(logN)^2)", "w0 + w1*K + w2*(K*N^(2)*logN)", "w0 + w1*K + w2*(K*N^(w3)*logN)"]

    equations = {
        'dt_lower' : '', 'dt_avg' : 'a + b*K*N*(logN)^2', 'dt_upper' : '',
        'rf_lower' : '', 'rf_avg' : '', 'rf_upper' : '',
        'sgd_lower' : '', 'sgd_avg' : '', 'sgd_upper' : ''

    }

    # likelihood function has multiple versions of eqs. here define which ones will be used.
    used_eq = [
        # 'dt_lower',
        'dt_avg',
        # 'dt_upper',
        # 'rf_lower', 'rf_avg', 'rf_upper',
        # 'sgd_lower', 'sgd_avg', 'sgd_upper'
    ]

    table = pd.DataFrame()
    train_folder_path, training_data_sets = datasets_names(folder_path='runtimes/all_dt')

    # find and load training data files
    for data_set in training_data_sets:
        print("dataset", data_set)
        x_data, y_data = load_dataset(data_set, train_folder_path)
        # x_train, y_train, data_name_train = load_training_data()

        # sys.exit()
        total_data_size = len(y_data)
        print("total_data_size:", total_data_size)

        y_predicted = list()

        x_train, x_test, y_train, y_test = process_train_data(x_data, y_data, 0.19, 0)
        # fit the model
        trace_pymc = mcmc_fit(x_train, y_train, used_eq)

        # x1_features, x2_size, y_runtime = load_data_pred(data_name_pred, test_folder_path)
        x1_features = x_test[:, 1]
        x2_size = x_test[:, 0]
        y_runtime = y_test
        hp_values = []

        for equation in used_eq:
            model = Model()
            with model:
                # load the model from pickle file
                trace = pickle.load(open("results/model_" + str(equation) + ".pickle", "rb"))

            mu_alpha = np.average(np.array(trace.get_values('alpha')))
            mu_beta = np.average(np.array(trace.get_values('beta')))
            mu_ceta = np.average(np.array(trace.get_values('ceta')))
            hp_values = [mu_alpha, mu_beta, mu_ceta]
            # predict on data_name_pred
            y_pred, y_pred_upper, y_pred_lower = mcmc_predict(trace, x1_features, x2_size, equation)
            y_predicted.append([y_pred, y_pred_upper, y_pred_lower, equations[equation]])

        # plot data
        # table = pd.concat([table, plot_prediction(x1_features, x2_size, y_runtime, y_predicted, data_set, data_set)])
        table = pd.concat([table, plot_get_hp_table(x1_features, x2_size, y_runtime, y_predicted, hp_values,
                                                    data_set, data_set)])
        y_predicted = list()

    table.to_html("dt_avg_19.html")
    print("end")
    # pdb.set_trace()



