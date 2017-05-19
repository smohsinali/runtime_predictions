import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


sns.set(font_scale=1.75)


def hp_analysis(algo_datasize):

    file_path = 'alphabeta/' + algo_datasize
    data = np.loadtxt(file_path, skiprows=1)

    alpha = data[:,0]
    # beta = data[:,1]
    beta = np.log(data[:,1])

    df = pd.DataFrame({ 'alpha': alpha,
                        'beta': beta,
                        })

    fig, ax = plt.subplots()
    ax.scatter(alpha, beta, s=50, cmap=plt.cm.coolwarm)
    ax.errorbar(alpha, beta, linestyle='None', capsize=5, capthick=2)
    # ax.scatter(np.log(true), np.log(predicted))
    # lims = [
    #     np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    #     np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    # ]

    # ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)


    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$log \beta$')
    # plt.xlim(0, 0.7)
    # plt.ylim(0, 0.7)

    # ax.title("Decision Tree Model: a + bKN(log(N))^2")
    plt.show()

    # print(scatter_plot + " not found")

# algo_datasize = 'rf_scatter_19.np'
algo_datasize = 'dt_28_ab.np'

#
# num_classes = np.loadtxt('scatterplot/num_classes.np', skiprows=1, dtype={'names':('ds_name', 'num_classes'),
#                                                                           'formats':('|S15', np.int )})
hp_analysis(algo_datasize=algo_datasize)