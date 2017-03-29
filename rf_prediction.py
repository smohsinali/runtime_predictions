import numpy as np
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
from data import load_test_data, load_training_data, datasets_names, load_dataset, process_train_data
from sklearn.ensemble import RandomForestRegressor
from plotting import plot_rf_results


if __name__ == "__main__":

    train_folder_path = 'runtimes'
    x_train, y_train, data_name_train = load_training_data(folder_path=train_folder_path)

    total_data_size = len(y_train)
    print("total_data_size:", total_data_size)

    y_predicted = list()

    test_folder_path = "runtimes/test/dt_graphs"
    files = next(os.walk(test_folder_path))[2]
    test_data_sets = sorted(list(set([x[16:-3] for x in files if re.search('np', x)])))

    clf = RandomForestRegressor(n_estimators=1300)
    clf.fit(x_train, y_train)

    table = pd.DataFrame()

    for dataset_test in test_data_sets:
        # load dataset on which prediction of runtimes is desired
        x1_features, x2_size, y_runtime = load_test_data(dataset_test, test_folder_path)
        x_test = np.array([x1_features, x2_size]).T

        y_predicted = clf.predict(x_test)
        table = pd.concat([table, plot_rf_results(x1_features, x2_size, y_runtime, y_predicted, dataset_test)])

    table.to_html("rf_predictions_dt.html")
    print("end")