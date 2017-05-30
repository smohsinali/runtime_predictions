import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


sns.set(font_scale=1.75)


def scatter_plot(algo_datasize, num_classes):
    """
    plots a scatter plot comparing predicted vs true runtimes
    :param algo_datasize: file that contains the data
    :param num_classes: array that contains number of classes of each dataset
    :return:
    """

    file_path = 'scatterplot/' + algo_datasize
    data = np.loadtxt(file_path, skiprows=1)

    true = data[:,0]
    predicted = data[:,1]
    uncertainity = data[:,2]
    classes = num_classes

    errors = predicted/100 * uncertainity

    df = pd.DataFrame({ 'true': true,
                        'predicted': predicted,
                        'uncertainty': uncertainity,
                        'classes': classes})

    fig, ax = plt.subplots()
    ax.scatter(true, predicted, s=50, c=classes, cmap=plt.cm.coolwarm)
    ax.errorbar(true, predicted, yerr=errors, linestyle='None', capsize=5, capthick=2)
    # ax.scatter(np.log(true), np.log(predicted))
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)


    plt.xlabel('True (Time/s)')
    plt.ylabel('Predicted (Time/s)')
    # plt.xlim(0, 0.7)
    # plt.ylim(0, 0.7)

    # ax.title("Decision Tree Model: a + bKN(log(N))^2")
    plt.show()

    # print(scatter_plot + " not found")

algo_datasize = 'dt_scatter_28.np'
# algo_datasize = 'rf_scatter_13.np'
# algo_datasize = 'rf_scatter_19.np'
# algo_datasize = 'rf_scatter_28.np'
# algo_datasize = 'sgd_scatter_13.np'
# algo_datasize = 'sgd_scatter_19.np'
# algo_datasize = 'sgd_scatter_28.np'

num_classes = np.loadtxt('scatterplot/num_classes.np', skiprows=1, dtype={'names':('ds_name', 'num_classes'),
                                                                          'formats':('|S15', np.int )})
scatter_plot(algo_datasize=algo_datasize, num_classes=num_classes['num_classes'])