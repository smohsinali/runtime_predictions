import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


sns.set(font_scale=1.75)

scatter_plot = 'dt_scatter'



if scatter_plot == 'dt_scatter':

    data = np.loadtxt('scatterplot/dt_scatter', skiprows=1)

    true = data[:,0]
    predicted = data[:,1]
    uncertainity = data[:,2]
    classes = data[:,3]

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


    plt.xlabel('True')
    plt.ylabel('Preicted')
    plt.xlim(0, 0.7)
    plt.ylim(0, 0.7)

    # ax.title("Decision Tree Model: a + bKN(log(N))^2")
    plt.show()


else:
    print(scatter_plot + " not found")
