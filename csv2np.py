
import os
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# this script takes all the csv data files from inside 'datasets' folder and converts them to np arrays and put them
# in "runtimes" folder.
if __name__ == "__main__":

    sgd_graphs = list()
    dt_graphs = list()
    rf_graphs = list()
    test = "_test/"
    # test = "/"
    # path = "datasets/2016_runtime_data_sparse" + test
    # path = "datasets/randforest" + test
    # path = "datasets/allData" + test
    # path = "datasets/all_rf" + test

    path = "datasets/allData/0/test/"
    train = 1 if test == "/" else 0

    print(path)
    files = next(os.walk(os.path.abspath(path)))[2]
    for i in range(len(files)):
        files[i] = path + files[i]

        # print(files)
        # curr_file = "2016_runtime_data/results_3_bac.csv"
        curr_file = files[i]

        # {#features, #data, runtimes_list} this dictionary combines similar runs and store thier individual runtimes
        # in runtime_list
        unique_data = dict()

        # create 2 different ids for dt and sgd
        ids = list()

        # keep track of different data sizes(later used to find max data size for normalization)
        num_data = list()

        with open(curr_file) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # create unique id for similar runs
                id = row["Name"] + "__" + row["Model"] + "__" + row["#Data"]
                # create unique ids for sgd and dt
                ids.append(row["Name"] + "__" + row["Model"])

                data = list()
                runtimes = list()
                num_data.append(float(row["#Data"]))

                if float(row["Runtime"]) < 0:
                    # print(id, float(row["Runtime"]))
                    continue

                if id not in unique_data:
                    runtimes.append(float(row["Runtime"]))
                    data = [float(row["#Feat"]), float(row["#Data"]), runtimes]
                    unique_data.update({id: data})
                else:
                    unique_data[id][2].append(float(row["Runtime"]))

        # add mean and std to unique_data based on runtimes_list. now unique_data looks like:
        # {#features, #data, runtimes_list, mean(runtime_list), std(runtime_list)}
        for ud in unique_data:
            unique_data[ud].append([np.mean(unique_data[ud][2]), np.std(unique_data[ud][2])])

        # print("numdata:", num_data, np.max(num_data))
        numData = np.max(num_data)

        ids = set(ids)
        # unique combines runs of similar models
        unique = dict()
        for id in ids:
            for uid in unique_data:
                if str(id) == str(uid)[0:len(str(id))]:
                    d = list()
                    # {#features, #data, mean(runtime_list), std(runtime_list)}
                    # data = [unique_data[uid][0], unique_data[uid][1]/numData, unique_data[uid][3][0], unique_data[uid][3][1]]
                    data = [unique_data[uid][0], unique_data[uid][1], unique_data[uid][3][0], unique_data[uid][3][1]]
                    if id not in unique:
                        d.append(data)
                        unique.update({id: [d]})
                    else:
                        unique[id][0].append(data)

            unique[id][0] = sorted(unique[id][0], key=itemgetter(1))

            data_ratio = [unique[id][0][i][1] for i in range(len(unique[id][0]))]
            num_features = [unique[id][0][i][0] for i in range(len(unique[id][0]))]
            mean_runtimes = [unique[id][0][i][2] for i in range(len(unique[id][0]))]

            # data_ratio = [np.log(unique[id][0][i][1]) for i in range(len(unique[id][0]))]
            # num_features = [np.log(unique[id][0][i][0]) for i in range(len(unique[id][0]))]
            # mean_runtimes = [np.log(unique[id][0][i][2]) for i in range(len(unique[id][0]))]

            if (str(id)[-2:] == "gd"):
                sgd_graphs.append([num_features, data_ratio, mean_runtimes, id])
            elif (str(id)[-2:] == "dt"):
                dt_graphs.append([num_features, data_ratio, mean_runtimes, id])
            elif (str(id)[-2:] == "rf"):
                rf_graphs.append([num_features, data_ratio, mean_runtimes, id])
            else:
                print("unknown model:", id)
                sys.exit()

    print("Putting all data in arrays")

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.ion()
    y_format = "%0.7f"
    # sgd = [#features, datasize, runtime]
    algorithms = [dt_graphs, rf_graphs, sgd_graphs]
    algo_string = ["dt_graphs", "rf_graphs", "sgd_graphs"]
    for a, algo in enumerate(algorithms):
        if len(algo) == 0:
            print('%s is empty' % algo_string[a])
            continue
        else:
            x_features = list()
            x_datasize = list()
            y_runtime = list()
            data_name = list()

            x_features_summary = list()
            x_datasize_summary = list()
            y_runtime_summary = list()
            data_name_summary = list()

            for dataset in algo:
                surf = ax.plot(dataset[1], dataset[0], dataset[2])

                x_datasize.extend(dataset[1])
                x_features.extend(dataset[0])
                y_runtime.extend(dataset[2])
                data_name.extend(dataset[3])

                x_datasize_summary.append(str(dataset[1][-1]))
                x_features_summary.append(str(dataset[0][-1]))
                y_runtime_summary.append(str(dataset[2][-1]))
                data_name_summary.append(str(dataset[3]))

                x_tmp = np.column_stack((dataset[1], dataset[0]))
                x_tmp_csv = np.column_stack((dataset[1], dataset[0], dataset[2]))
                y_tmp = np.array(dataset[2])
                y_tmp = y_tmp[:, np.newaxis]

                if test == "_test/":
                    np.savetxt("runtimes/test/" + algo_string[a] + "/x_runtime_train_" + dataset[3] + ".np", x_tmp, fmt="%0.1f")
                    np.savetxt("runtimes/test/" + algo_string[a] + "/y_runtime_train_" + dataset[3] + ".np", y_tmp, fmt=y_format)
                    # np.savetxt("runtimes/test/" + algo_string[a] + "/y_runtime_train_" + dataset[3] + ".csv", x_tmp_csv,
                    #            delimiter=",", fmt=["%d", "%d", "%0.6f"], header='#Data,#Feat,Runtime')
                # if train == 1:
                #     np.savetxt("runtimes/train/" + algo_string[a] + "/x_runtime_train_" + dataset[3] + ".np", x_tmp, fmt="%0.1f")
                #     np.savetxt("runtimes/train/" + algo_string[a] + "/y_runtime_train_" + dataset[3] + ".np", y_tmp, fmt=y_format)
                #     # np.savetxt("runtimes/train/" + algo_string[a] + "/y_runtime_train_" + dataset[3] + ".csv",x_tmp_csv,
                #     #            delimiter=",", fmt=["%d", "%d", "%0.6f"], header='#Data,#Feat,Runtime')

            x = np.column_stack((x_datasize, x_features))
            y = np.array(y_runtime)
            y = y[:, np.newaxis]
            x_csv = np.column_stack((x_datasize, x_features, y_runtime))
            x_summary = np.column_stack((data_name_summary, x_datasize_summary, x_features_summary, y_runtime_summary))

            # y = np.column_stack((data_name, y_runtime))
            if test == "_test/":
                print('no combined file for the tests')
                # np.savetxt("runtimes/x_runtime_train_allTest_" + algo_string[a] + ".np", x, fmt="%0.1f")
                # np.savetxt("runtimes/y_runtime_train_allTest_" + algo_string[a] + ".np", y, fmt=y_format)
                # # np.savetxt("runtimes/y_runtime_train_allTest_" + algo_string[a] + ".csv", x_csv, delimiter=',',
                # #            fmt=["%d", "%d", "%0.6f"], header='#Data,#Feat,Runtime')
                # np.savetxt("runtimes/" + algo_string[a] + "_summary.csv", x_summary, delimiter=',',
                #            fmt=["%s","%d", "%d", "%0.6f"], header='Name,#Data,#Feat,Runtime')

            else:
                np.savetxt("runtimes/x_runtime_train_allTrain_" + algo_string[a] + ".np", x, fmt="%0.1f")
                np.savetxt("runtimes/y_runtime_train_allTrain_" + algo_string[a] + ".np", y, fmt=y_format)
                # np.savetxt("runtimes/y_runtime_train_allTest_" + algo_string[a] + ".csv", x_csv, delimiter=',',
                #            fmt=["%d", "%d", "%0.6f"], header='#Data,#Feat,Runtime')
                # np.savetxt("runtimes/" + algo_string[a] + "_summary.csv", x_summary, delimiter=',',
                #             fmt=["%s", "%s", "%s", "%s"],
                #             header='Name,#Data,#Feat,Runtime')

            print("done putting data in arrays, saved in .np files")
            # ax.zaxis.set_major_locator(LinearLocator(10))
            # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

            # fig.colorbar(surf, shrink=0.5, aspect=5)
            # ax.set_zscale('log')
            # ax.set_zlim(0, 100)
            # ax.set_zlim(0, 2)
            # if algo_string[a] == "rf_graphs":
            if 1 == 2:
                print("plotting data")
                ax.set_xlabel("data")
                ax.set_ylabel("features")
                ax.set_zlabel("runtimes")
                ax.set_title(algo_string[a])
                plt.show()
                # plt.savefig(algo_string[a] + ".png", dpi=150)
                input()
                ax.clear()

            print("exiting")
            # print(row['name'], row['model'], row['#data'], row['#features'], row['runtime'])