import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pymc3 import Model, Normal, HalfNormal
from pymc3 import find_MAP
from pymc3 import NUTS, sample, Metropolis
from pymc3 import traceplot
from scipy import optimize


def likelihood_knlogn(w, k, n, equation):
    val = 'not defined'
    if equation == 'dt_lower':
        # a + bN(log(N))^2
        n_log = np.log2(n) ** 2

        val = w[0] + w[1] * (n * n_log)

    if equation == 'dt_avg':
        # a + bkn logn
        n_log = np.log2(n) ** 2

        val = w[0] + w[1] * (k * n * n_log)

    if equation == 'dt_upper':
        # a + bkN^2(log(N))
        n_log = np.log2(n)
        n_mod = n ** 2

        val = w[0] + w[1] * (k * n_mod * n_log)

    if equation == 'rf_lower':
        # 'a + bN*sqrt(K)*(log(N))^2'
        n_log = np.log2(n)**2
        k_mod = np.sqrt(k)
        val = w[0] + w[1] * k_mod * n * n_log

    if equation == 'rf_lower_2':
        # 'a + bN*sqrt(K)*(log(N))^2'
        n_log = np.log2(n) ** 2
        k_mod = np.sqrt(k)
        val = w[0] + w[1]  * n * n_log + w[2] * k_mod

    if equation == 'rf_avg':
        # an + bkn (logn)^2
        n_log = (np.log2(n))**2
        n_mod = n ** 1
        val = w[0] +  w[1] * (k * n_mod * n_log)
        # val = w[0] * n_mod + w[1] * k_mod

    # if equation == 'rf_upper':
    #     # an + bkn^1.2*(logn)^2
    #     n_log = (np.log2(n))**2
    #     n_mod = n ** 1.2
    #     val = w[0] + w[1] * (k * n_mod * n_log)
    if equation == 'rf_upper':
        # a + bn(logn)^2
        n_log = (np.log2(n)) ** 2
        n_mod = n ** 1
        val = w[0] + w[1]  * n * n_log

    if equation == 'rf_upper_2':
        # a + bn(logn)^2
        n_log = (np.log2(n)) ** 2
        n_mod = n
        val = w[0] + w[1] * k * n_log

    if equation == 'sgd_lower':
        # a+bn
        n_log = (np.log2(n))
        # n_mod = n ** 1.2
        val = w[0] + w[1]*k*n

    if equation == 'sgd_lower_2':
        # a + bn^1.3
        val = w[0] + w[1] * n**1.3


    if equation == 'sgd_avg':
        # a + bn + ck
        val = w[0]  + w[1] * n +  w[2] * k

    if equation == 'sgd_upper':
        # a + bn^1.2 + ck
        n_mod = n ** 1.2

        val = w[0] + w[1] * n_mod + w[2] * k
    #
    # if equation == 'sgd_lower':
    #     # an^1.2 + bk + c
    #     n_log = (np.log2(n))**2
    #
    #     val = w[0] * n  * n_log + w[1] * k

    if val == 'not defined':
        print(equation + ' not defined')
    else:
        return val


# defining the model with given params
def mcmc_model(no_of_features, data_size, training_time, equation):

    basic_model = Model()
    with basic_model:

        # ## Priors for unknown model params
        alpha = HalfNormal('alpha', sd=1)
        beta = Normal('beta', mu=0, sd=1)
        ceta = Normal('ceta', mu=1.2, sd=1)
        # gamma = Normal('gamma', mu=0, sd=.5)
        sigma = HalfNormal('sigma', sd=1)
        # sigma = Normal('sigma', mu=0, sd=1)

        params = (alpha, beta, ceta)
        # ## Expected value of outcome
        mu = likelihood_knlogn(params, no_of_features, data_size, equation)

        # ## Likelihood of obs
        y_obs = Normal('Y_obs', mu=mu, sd=sigma, observed=training_time)

        # ## obtain starting values via MAP
        start = find_MAP(fmin=optimize.fmin_powell)

        # ## draw posterior samples

        # using NUTS
        # trace = sample(50, start=start)

        # using metropolis hastings with 2000 burin steps
        step1 = Metropolis([alpha, beta, ceta, sigma])
        sample(10000, start=start, step=step1)
        trace = sample(20000, start=start, step=step1)

    print("mcmc_model end")
    return trace


def mcmc_fit(xdata, ytime, eq_used, data_set):

    print("length xdata:", len(xdata), "\nlength ytime:", len(ytime))
    trace = [] # [y_cal, y_cal_upper, y_cal_lower, size]

    for equation in eq_used:

        # creating mcmc model
        trace = mcmc_model(xdata[:, 1], xdata[:, 0], ytime, equation)

        # save the learned mode in pickle file
        pickle.dump(trace, open("results/model_" + str(equation) + ".pickle", "wb"), protocol=-1)

        traceplot(trace)
        # move plot about learned param values to results folder
        os.rename("param.png", "results/mcmc_" + data_set + "_" + str(equation) + ".png")
        plt.close()

    print("mcmc_fit finish")
    return trace

# predict values for y
def mcmc_predict(trace, no_of_features, data_size, equation):

    standard_dev = 1

    alpha = np.array(trace.get_values('alpha'))
    mu_alpha = np.average(alpha)
    std_alpha = standard_dev * np.std(alpha)

    beta = np.array(trace.get_values('beta'))
    mu_beta = np.average(beta)
    std_beta = standard_dev * np.std(beta)

    ceta = np.array(trace.get_values('ceta'))
    mu_ceta = np.average(ceta)
    std_ceta = standard_dev * np.std(ceta)
    #
    # gamma = np.array(trace.get_values('gamma'))
    # mu_gamma = np.average(gamma)
    # std_gamma = standard_dev * np.std(gamma)

    # params = (mu_alpha, mu_beta, mu_ceta, mu_gamma)
    # params_upper = (mu_alpha + std_alpha, mu_beta + std_beta, mu_ceta + std_ceta, mu_gamma + std_gamma)
    # params_lower = (mu_alpha - std_alpha, mu_beta - std_beta, mu_ceta - std_ceta, mu_gamma - std_gamma)

    params = (mu_alpha, mu_beta, mu_ceta)
    params_upper = (mu_alpha + std_alpha, mu_beta + std_beta, mu_ceta + std_ceta)
    params_lower = (mu_alpha - std_alpha, mu_beta - std_beta, mu_ceta - std_ceta)
    #
    # params = (mu_alpha, mu_beta)
    # params_upper = (mu_alpha + std_alpha, mu_beta + std_beta)
    # params_lower = (mu_alpha - std_alpha, mu_beta - std_beta)

    y_pred = likelihood_knlogn(params, no_of_features, data_size, equation)
    y_pred_upper = likelihood_knlogn(params_upper, no_of_features, data_size, equation)
    y_pred_lower = likelihood_knlogn(params_lower, no_of_features, data_size, equation)

    return y_pred, y_pred_upper, y_pred_lower

