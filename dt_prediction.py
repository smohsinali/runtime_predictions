import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import theano
from pymc3 import Model, Normal, HalfNormal
from pymc3 import find_MAP
from pymc3 import NUTS, sample, Metropolis, Slice
from pymc3 import traceplot
from scipy import optimize
from sklearn.cross_validation import train_test_split


def likelihood_knlogn(w0, w1, w2, k, n, base):
    n_mod = n ** 1
    k_mod = k ** 1

    # if base == 1:
    #     val = w0 + w1 * (k_mod * n_mod * np.log(n))
    if base == 2:
        # val = w0 + w1 * (k_mod * n_mod * np.log2(n))
        # print(val)
        val = w0 + w1 * (k_mod * n_mod * np.log2(n)) + w2 * (k_mod * n_mod ** 2 * np.log2(n))
    # elif base == 10:
    #     val = w0 + w1 * (k_mod * n_mod * np.log10(n))

    if hasattr(val, 'name'):
        return val

    else:
        x = [i for i in range(len(val)) if val[i] < 0]
        val[x] = 0
        return val


# defining the model with given params
def mcmc_model(parameters):

    x1 = parameters[0]
    x2 = parameters[1]
    y = parameters[2]
    base = parameters[3]

    basic_model = Model()
    with basic_model:

        # Priors for unknown model params
        alpha = Normal('alpha', mu=0, sd=1)
        beta = Normal('beta', mu=0, sd=1, shape=1)
        ceta = Normal('ceta', mu=0, sd=1, shape=1)

        sigma = HalfNormal('sigma', sd=1)

        # Expected value of outcome
        # mu = alpha + beta * (x1 * x2 ** 2 * np.log2(x2))
        mu = likelihood_knlogn(alpha, beta, ceta, x1, x2, base)

        # Likelihood of obs
        y_obs = Normal('Y_obs', mu=mu, sd=sigma, observed=y)

        # obtain starting values via MAP
        start = find_MAP(fmin=optimize.fmin_powell)

        # step1 = Metropolis(vars=[alpha])
        # step1 = NUTS(vars=[alpha, beta])
        # step2 = Metropolis(vars=[ceta, sigma])
        # draw posterior samples
        trace = sample(30, start=start)
        # trace = sample(50, start=start, step=step)

    print("mcmc_model end")
    return trace


# predict values for y
def mcmc_predict(trace, x1_f, x2_f, base):
    alpha = np.array(trace.get_values('alpha'))
    mu_alpha = np.average(alpha)
    std_alpha = 3 * np.std(alpha)

    beta = np.array(trace.get_values('beta'))
    mu_beta = np.average(beta)
    std_beta = 3 * np.std(beta)

    ceta = np.array(trace.get_values('ceta'))
    mu_ceta = np.average(ceta)
    std_ceta = 1 * np.std(ceta)

    y_cal = likelihood_knlogn(mu_alpha, mu_beta, mu_ceta, x1_f, x2_f, base)
    y_cal_upper = likelihood_knlogn(mu_alpha + std_alpha, mu_beta + std_beta, mu_ceta + std_ceta, x1_f, x2_f, base)
    y_cal_lower = likelihood_knlogn(mu_alpha - std_alpha, mu_beta - std_beta, mu_ceta - std_ceta, x1_f, x2_f, base)

    return y_cal, y_cal_upper, y_cal_lower


def mcmc_fit(xdata, ytime):

    print("length xdata:", len(xdata))
    print("length ytime:", len(ytime))
    base = 2
    params = [xdata[:, 1], xdata[:, 0], ytime, base]

    # creating mcmc model
    trace = mcmc_model(params)
    traceplot(trace)
    plt.close()
    # [y_cal, y_cal_upper, y_cal_lower, size]
    return trace


def plot(x, y, y_calculated, data_name, data_name_pred):

    plt.plot(x, y, label="non-predicted")
    for y in y_calculated:
        y_cal = y[0]
        y_cal_upper = y[1]
        y_cal_lower = y[2]
        plt.plot(x, y_cal, label=y[3])
        # plt.fill_between(x, y_cal_upper, y_cal_lower, facecolor='yellow', alpha=0.2)

    plt.xlabel("Number of Instances")
    plt.ylabel("Time(s)")
    plt.title("Dataset:" + data_name + "\nPrediction using " + data_name_pred)
    plt.legend(loc='best')
    plt.savefig("results/mc_" + data_name + "_pred_highres.png", dpi=300)
    plt.savefig("results/mc_" + data_name + "_pred.png", dpi=200)


def load_data():
    name = next(os.walk(os.path.abspath("runtimes/")))[2]
    names = sorted(list(set([x[16:-3] for x in name if re.search('np', x)])))
    sets = dict()
    for i, j in enumerate(names):
        print(i, ":", j)
        sets[i] = j
    print("Select Dataset which will be used for predicting runtimes.")
    ids = input("Enter data id:")
    if re.search("[A-Za-z]", ids):
        print("Error: Please enter id in numbers only")
        sys.exit()
    ids = int(ids)
    if ids < 0 or ids >= len(names):
        print("Wrong id: no data set with this id exist")
        sys.exit()

    files = [x for x in name if re.search(sets[ids] + ".np", x)]

    if len(files) < 2:
        print("apparently one of files is missing, please investigate")
        sys.exit()

    x_path = os.path.join("runtimes/", list(filter(lambda x: x.startswith("x"), files))[0])
    y_path = os.path.join("runtimes/", list(filter(lambda x: x.startswith("y"), files))[0])
    print("Loading following data files for dataset " + sets[ids] + ":\n", x_path, y_path)
    x_runtime = np.loadtxt(x_path)
    y_runtime = np.loadtxt(y_path)
    y_runtime = np.around(y_runtime, decimals=4)

    return x_runtime, y_runtime, sets[ids]

if __name__ == "__main__":
    # find and load data files
    x_data, y_data, data_name_pred = load_data()
    # x_data = x_data[0:89]
    # y_data = y_data[0:89]
    # x_runtime = x_data
    # y_runtime = y_data
    # for testing only (refactor later)

    data_name = input("Enter Dataset Name which will be used for plotting reference time:")
    x_runtime = np.loadtxt("runtimes/x_runtime_train_" + data_name + ".np")
    y_runtime = np.loadtxt("runtimes/y_runtime_train_" + data_name + ".np")
    y_runtime = np.around(y_runtime, decimals=4)
    x1_f = x_runtime[:, 1]
    x2_f = x_runtime[:, 0]
    base = 2

    total_data_size = len(y_data)
    print("total_data_size:", total_data_size)
    a = 6
    # a = [20, 30]
    y_calculated = list()
    for i in range(4):
        # size of data used for training
        a = int(a * 1.5)
        # data_size = total_data_size * (a / 100)
        ratio = a / 100.0
        print("ratio:", ratio)

        # find samples using mcmc
        # learn on x_data
        # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
        #                                                     train_size=ratio, random_state=random.randint(1, 100))
        x_train = x_data[0:a]
        y_train = y_data[0:a]
        trace = mcmc_fit(x_train, y_train)

        # predict on data_name
        y_cal, y_cal_upper, y_cal_lower = mcmc_predict(trace, x1_f, x2_f, base)
        y_calculated.append([y_cal, y_cal_upper, y_cal_lower, a])

    # plot data
    plot(x2_f, y_runtime, y_calculated, data_name, data_name_pred)








