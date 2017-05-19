import numpy as np
import seaborn as sns
import pandas as pd


sns.set(font_scale=1.75)
# bx_plot = 'dt_datasize'
# bx_plot = 'dt_models'
# bx_plot = 'rf_datasize'
bx_plot = 'rf_models'
# bx_plot = 'sgd_datasize'
# bx_plot = 'sgd_models'


if bx_plot == 'dt_datasize':

    data = np.loadtxt('boxplots/boxplot_dt_datasize_neg.np', skiprows=1)
    # data = np.loadtxt('boxplots/boxplot_dt_datasize.np')

    bx1 = data[:,0]
    bx2 = data[:,1]
    bx3 = data[:,2]
    bx4 = data[:,3]

    df = pd.DataFrame({ '9%': bx1,
                        '13%': bx2,
                        '19%': bx3,
                        '28%': bx4})

    order = ["9%", "13%", "19%", "28%"]
    sns.swarmplot(data=df, order=order, color='black')
    sns.boxplot(data=df, order=order)

    sns.plt.xlabel('Data used')
    sns.plt.ylabel('Difference to True value(%)')
    sns.plt.ylim(-55, 95)

    sns.plt.title(r"Decision Tree Model: $\alpha + \beta k Nlog^2N$")
    sns.plt.show()


elif bx_plot == 'rf_datasize':

    data = np.loadtxt('boxplots/boxplot_rf_datasize_neg.np', skiprows=1)

    bx1 = data[:,0]
    bx2 = data[:,1]
    bx3 = data[:,2]
    bx4 = data[:,3]

    df = pd.DataFrame({ '9%': bx1,
                        '13%': bx2,
                        '19%': bx3,
                        '28%': bx4})

    order = ["9%", "13%", "19%", "28%"]
    sns.boxplot(data=df, order=order)
    # sns.swarmplot(data=df, order=order, color='black')
    sns.plt.xlabel('Data used')
    sns.plt.ylabel('Difference to True value(%)')
    sns.plt.ylim(-130, 130)

    sns.plt.title(r"Random Forests Model: $\alpha + \beta Mp\widetilde{N}log^2\widetilde{N}$")
    sns.plt.show()

if bx_plot == 'sgd_datasize':

    data = np.loadtxt('boxplots/boxplot_sgd_datasize_neg.np', skiprows=1)

    bx1 = data[:,0]
    bx2 = data[:,1]
    bx3 = data[:,2]
    bx4 = data[:,3]

    df = pd.DataFrame({ '9%': bx1,
                        '13%': bx2,
                        '19%': bx3,
                        '28%': bx4})

    order = ["9%", "13%", "19%", "28%"]
    sns.boxplot(data=df, order=order)
    # sns.swarmplot(data=df, order=order, color='black')
    sns.plt.xlabel('Data used')
    sns.plt.ylabel('Difference to True value(%)')
    sns.plt.ylim(0, 130)

    sns.plt.title(r"SGD Model: $\alpha + \beta N$")
    sns.plt.show()

elif bx_plot == 'dt_models':
    data = np.loadtxt('boxplots/boxplot_dt.np', skiprows=1)

    bx1 = data[:,0]
    bx2 = data[:,1]
    bx3 = data[:,2]
    bx4 = data[:,3]

    eq1 = r"$\alpha k + \beta Nlog^2N$"
    # eq2 = "a + bKN(logN)^2"
    eq2 = r"$\alpha + \beta kNlog^2N$"
    # eq3 = "a + bKN^2(log(N))"
    eq3 = r"$\alpha + \beta kN^2logN$"
    # eq4 = "a + bN(log(N))^2"
    eq4 = r"$\alpha + \beta Nlog^2N$"

    df = pd.DataFrame({
        eq1: bx1,
        eq2: bx2,
        eq3: bx3,
        eq4: bx4
    })

    order = [ eq2, eq4, eq3]
    # sns.swarmplot(data=df, order=order, color='black')
    sns.boxplot(data=df, order=order)
    sns.plt.xlabel('Models')
    sns.plt.ylabel('Difference to True value(%)')
    # sns.plt.ylim(0, 130)
    sns.plt.title("Decision Trees Prediction Accuracy Overview (28% data used)")
    sns.plt.show()

elif bx_plot == 'rf_models':

    data = np.loadtxt('boxplots/boxplot_rf_new.np', skiprows=1)

    bx1 = data[:,0]
    bx2 = data[:,1]
    bx3 = data[:,2]
    bx4 = data[:,3]
    bx5 = data[:,4]


    eq1 = 'a + b(sqrt(K))N(log(N))^2'
    eq2 = 'a + bN(log(N))^2 + c(sqrt(k))'
    eq3 = r'$\alpha + \beta Mp\widetilde{N}log^2\widetilde{N}$'
    eq4 = r'$\alpha + \beta \widetilde{N}log^2\widetilde{N}$'
    eq5 = r'$\alpha + \beta Mp\widetilde{N}^2log\widetilde{N}$'

    df = pd.DataFrame({ eq1: bx1,
                        eq2: bx2,
                        eq3: bx3,
                        eq4: bx4,
                        eq5: bx5,
                        })

    order = [eq3, eq4, eq5]
    sns.swarmplot(data=df, order=order, color='black')
    sns.boxplot(data=df, order=order)
    sns.plt.xlabel('Models')
    sns.plt.ylabel('Difference to True value(%)')
    sns.plt.title("Random Forests Prediction Accuracy Overview (28% data used)")
    sns.plt.show()


elif bx_plot == 'sgd_models':

    data = np.loadtxt('boxplots/boxplot_sgd.np', skiprows=1)

    bx1 = data[:,0]
    bx2 = data[:,1]
    bx3 = data[:,2]
    bx4 = data[:,3]
    bx5 = data[:,4]
    bx6 = data[:,5]
    bx7 = data[:,6]

    eq1 = 'aN + bK + c'
    eq2 = 'aN^1.2 + bK + c'
    eq3 = r'$\alpha + \beta kN$'
    eq4 = 'a + bN^1.3'
    eq5 = 'a + bKN^1.2'
    eq6 = 'a + bKN(log(N))'
    eq7 = r'$\alpha + \beta N$'


    df = pd.DataFrame({
        eq1: bx1,
        eq2: bx2,
        eq3: bx3,
        eq4: bx4,
        eq5: bx5,
        eq6: bx6,
        eq7: bx7
    })

    order = [eq3, eq7]
    sns.swarmplot(data=df, order=order, color='black')
    sns.boxplot(data=df, order=order)
    sns.plt.xlabel('Models')
    sns.plt.ylabel('Difference to True value(%)')
    sns.plt.ylim(-50, 50)
    sns.plt.title("SGD Prediction Accuracy Overview (28% data used)")
    sns.plt.show()

else:
    print(bx_plot + " not found")
