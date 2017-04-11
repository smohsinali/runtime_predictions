import numpy as np
import seaborn as sns
import pandas as pd


sns.set(font_scale=1.75)

# bx_plot = 'dt_datasize'
# bx_plot = 'dt_models'
# bx_plot = 'rf_datasize'
# bx_plot = 'rf_models'
# bx_plot = 'sgd_datasize'
bx_plot = 'sgd_models'


if bx_plot == 'dt_datasize':

    data = np.loadtxt('boxplots/boxplot_dt_datasize.np')

    bx1 = data[:,0]
    bx2 = data[:,1]
    bx3 = data[:,2]
    bx4 = data[:,3]

    df = pd.DataFrame({ '9%': bx1,
                        '13%': bx2,
                        '19%': bx3,
                        '28%': bx4})


    sns.boxplot(data=df, order=["9%", "13%", "19%", "28%"])
    sns.plt.xlabel('Data used')
    sns.plt.ylabel('Difference from True value(%)')
    sns.plt.ylim(0, 130)

    sns.plt.title("Decision Tree Model: a + bKN(log(N))^2")
    sns.plt.show()


elif bx_plot == 'rf_datasize':

    data = np.loadtxt('boxplots/boxplot_rf_datasize.np', skiprows=1)

    bx1 = data[:,0]
    bx2 = data[:,1]
    bx3 = data[:,2]
    bx4 = data[:,3]

    df = pd.DataFrame({ '9%': bx1,
                        '13%': bx2,
                        '19%': bx3,
                        '28%': bx4})


    sns.boxplot(data=df, order=["9%", "13%", "19%", "28%"])
    sns.plt.xlabel('Data used')
    sns.plt.ylabel('Difference from True value(%)')
    sns.plt.ylim(0, 130)

    sns.plt.title("Random Forests Model: a + bKN(log(N))^2")
    sns.plt.show()

if bx_plot == 'sgd_datasize':

    data = np.loadtxt('boxplots/boxplot_sgd_datasize.np', skiprows=1)

    bx1 = data[:,0]
    bx2 = data[:,1]
    bx3 = data[:,2]
    bx4 = data[:,3]

    df = pd.DataFrame({ '9%': bx1,
                        '13%': bx2,
                        '19%': bx3,
                        '28%': bx4})


    sns.boxplot(data=df, order=["9%", "13%", "19%", "28%"])
    sns.plt.xlabel('Data used')
    sns.plt.ylabel('Difference from True value(%)')
    sns.plt.ylim(0, 130)

    sns.plt.title("SGD Model: a + bN")
    sns.plt.show()

elif bx_plot == 'dt_models':
    data = np.loadtxt('boxplots/boxplot_dt.np', skiprows=1)

    bx1 = data[:,0]
    bx2 = data[:,1]
    bx3 = data[:,2]
    bx4 = data[:,3]

    eq1 = "aK + bN(logN)^2"
    eq2 = "a + bKN(logN)^2"
    eq3 = "a + bKN^2(log(N))"
    eq4 = "a + bN(log(N))^2"

    df = pd.DataFrame({
        eq1: bx1,
        eq2: bx2,
        eq3: bx3,
        eq4: bx4
    })

    order = [ eq2, eq4, eq3]
    sns.swarmplot(data=df, order=order, color='black')
    sns.boxplot(data=df, order=order)
    sns.plt.xlabel('Models')
    sns.plt.ylabel('Difference from True value(%)')
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
    eq3 = 'a + bKN(log(N))^2'
    eq4 = 'a + bN(log(N))^2'
    eq5 = 'a + bKN^2(log(N))'

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
    sns.plt.ylabel('Difference from True value(%)')
    sns.plt.title("Random Forest Prediction Accuracy Overview (28% data used)")
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
    eq3 = 'a + bkN'
    eq4 = 'a + bN^1.3'
    eq5 = 'a + bKN^1.2'
    eq6 = 'a + bKN(log(N))'
    eq7 = 'a + bN'


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
    sns.plt.ylabel('Difference from True value(%)')
    sns.plt.ylim(-50, 50)
    sns.plt.title("SGD Prediction Accuracy Overview (28% data used)")
    sns.plt.show()

else:
    print(bx_plot + " not found")
