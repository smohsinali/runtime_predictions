import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from pymc3 import Model, Normal, HalfNormal
from pymc3 import find_MAP
from pymc3 import NUTS, sample
from scipy import optimize


def likelihood_knlogn(w0, w1, k, n, base):
    n_mod = n ** 1
    k_mod = k ** 1

    if base == 1:
        val = w0 + w1 * (k_mod * n_mod * np.log(n))
    elif base == 2:
        val = w0 + w1 * (k_mod * n_mod * np.log2(n))
    elif base == 10:
        val = w0 + w1 * (k_mod * n_mod * np.log10(n))

    return val


def mcmc_fit(x_runtime, y_runtime, percent_data):
    # true param values
    alpha, sigma = 1, 1
    beta = 3
    print("training on %d percent data" % percent_data)

    # variables
    # length of runtime files is 90 (1% to 90% data)
    x1_f = x_runtime[:, 1]
    x2_f = x_runtime[:, 0]
    x1 = x_runtime[:percent_data, 1]
    x2 = x_runtime[:percent_data, 0]
    y = y_runtime[:percent_data]
    base = 2

    # define MCMC model
    basic_model = Model()
    with basic_model:

        # Priors for unknown model params
        alpha = Normal('alpha', mu=0, sd=10)
        beta = Normal('beta', mu=0, sd=10, shape=1)
        sigma = HalfNormal('sigma', sd=1)

        # Expected value of outcome
        # mu = alpha + beta * (x1 * x2 ** 2 * np.log2(x2))
        mu = likelihood_knlogn(alpha, beta, x1, x2, base)

        # Likelihood of obs
        y_obs = Normal('Y_obs', mu=mu, sd=sigma, observed=y)

        # obtain starting values via MAP
        start = find_MAP(fmin=optimize.fmin_powell)

        # draw 2000 posterior samples
        trace = sample(50, start=start)

    # summary(trace)
    alpha = np.array(trace.get_values('alpha'))
    mu_alpha = np.average(alpha)
    std_alpha = 1 * np.std(alpha)

    beta = np.array(trace.get_values('beta'))
    mu_beta = np.average(beta)
    std_beta = 1 * np.std(beta)

    y_cal = likelihood_knlogn(mu_alpha, mu_beta, x1_f, x2_f, base)
    y_cal_upper = likelihood_knlogn(mu_alpha + std_alpha, mu_beta + std_beta, x1_f, x2_f, base)
    y_cal_lower = likelihood_knlogn(mu_alpha - std_alpha, mu_beta - std_beta, x1_f, x2_f, base)

    sigma = np.array(trace.get_values('sigma'))

    print("\nmcmc_fit end")
    return [y_cal, y_cal_upper, y_cal_lower, percent_data]


def plot(x, y, y_calculated, data_name):

    plt.plot(x, y, label="non-predicted")
    for y in y_calculated:
        y_cal = y[0]
        y_cal_upper = y[1]
        y_cal_lower = y[2]
        plt.plot(x, y_cal, label=y[3])
        # plt.fill_between(x, y_cal_upper, y_cal_lower, facecolor='yellow', alpha=0.2)

    plt.xlabel("Number of Instances")
    plt.ylabel("Time(s)")
    plt.title("Dataset:" + data_name + "\nPrediction ")
    plt.legend(loc='best')
    plt.savefig("results/mc_" + data_name + "_pred.png", dpi=600)


def load_data():
    name = next(os.walk(os.path.abspath("runtimes/")))[2]
    names = sorted(list(set([x[16:-3] for x in name if re.search('np', x)])))
    sets = dict()
    for i, j in enumerate(names):
        print(i, ":", j)
        sets[i] = j

    ids = input("Enter data id:")
    if re.search("[A-Za-z]", ids):
        print("Error: Please enter id in numbers only")
        sys.exit()
    ids = int(ids)
    if ids < 0 or ids >= len(names):
        print("Wrong id: no data set with this id exist")
        sys.exit()

    files = [x for x in name if re.search(sets[ids], x)]

    if len(files) != 2:
        print("apparently one of files is missing, please investigate")
        sys.exit()

    x_path = os.path.join("runtimes/", list(filter(lambda x: x.startswith("x"), files))[0])
    y_path = os.path.join("runtimes/", list(filter(lambda x: x.startswith("y"), files))[0])
    print("Loading following data files for dataset " + sets[ids] + ":\n", x_path, y_path)
    x_runtime = np.loadtxt(x_path)
    y_runtime = np.loadtxt(y_path)

    return x_runtime, y_runtime, sets[ids]

if __name__ == "__main__":
    # find and load data files
    x_data, y_data, data_name = load_data()

    a = 2
    y_calculated = list()
    for i in range(10):
        a = int(a * 1.5)
        # find samples using mcmc
        y_calculated.append(mcmc_fit(x_data, y_data, a))

    # plot data
    plot(x_data[:, 0], y_data, y_calculated, data_name)










