import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
# from pymc3 import Model, Normal, HalfNormal
# from pymc3 import find_MAP
# from pymc3 import NUTS, sample, Metropolis
# from pymc3 import traceplot
import pymc as pm
from data import load_test_data, load_training_data, plot_predictions
# def likelihood_knlogn(w, k, n, equation):
#
#     if equation == 1:
#         n_log = np.log2(n) ** 2
#         n_mod = n
#         k_mod = k * 0
#
#         # val = w[0] + w[1] * k_mod + w[2] * (k * n_mod * n_log)
#         val = w[0] + w[1] * (k * n_mod * n_log)
#
#     if equation == 2:
#         n_log = np.log2(n)
#         n_mod = n ** 2
#         k_mod = k * 1
#         val = w[0] + w[1] * k_mod + w[2] * (k * n_mod * n_log)
#
#     if equation == 3:
#         n_log = np.log2(n)
#         n_mod = n
#         k_mod = k * 1
#         val = w[0] + w[1] * k_mod + w[2] * (k * n_mod**w[3] * n_log)
#
#     return val


# defining the model with given params
def mcmc_model(parameters):

    x_features = parameters[1]  # x_features
    x_size = parameters[0]  # x_size
    y = parameters[2]
    equation = parameters[3]

    alpha = pm.Normal('alpha', mu=0, tau=1, value=0)
    beta = pm.Normal('beta', mu=0, tau=1, value=0)
    # ceta = pm.Normal('ceta', mu=0, tau=1, value=0)

    # basic_model = Model()
    # with basic_model:
    #
    #     # ## Priors for unknown model params
    #     alpha = HalfNormal('alpha', sd=1)
    #     beta = Normal('beta', mu=0, sd=.00001)
    #     ceta = Normal('ceta', mu=0, sd=.0001)
    #     gamma = Normal('gamma', mu=0, sd=.5)
    #     sigma = HalfNormal('sigma', sd=1)
    #     # sigma = Normal('sigma', mu=0, sd=1)
    #
    #     params = (alpha, beta, ceta, gamma)
    #     # ## Expected value of outcome
    #     mu = likelihood_knlogn(params, x1, x2, equation)
    #
    #     # ## Likelihood of obs
    #     y_obs = Normal('Y_obs', mu=mu, sd=sigma, observed=y)
    #
    #     # ## obtain starting values via MAP
    #     start = find_MAP(fmin=optimize.fmin_powell)
    #
    #     # ## draw posterior samples
    #
    #     # using NUTS
    #     # trace = sample(50, start=start)
    #
    #     # using metropolis hastings with 2000 burin steps
    #     step1 = Metropolis([alpha, beta, ceta, gamma, sigma])
    #     sample(20000, start=start, step=step1)
    #     trace = sample(50000, start=start, step=step1)
    @pm.deterministic
    def likelihood_knlogn(n=x_size, k=x_features, alpha=alpha, beta=beta):
        n_log = np.log2(n)
        n_log_square = n_log ** 2

        val = alpha + beta + (k * n * n_log_square)

        return val

    observation = pm.Normal('obs', mu=likelihood_knlogn, value=y, observed=True)

    model = pm.Model([observation, likelihood_knlogn, alpha, beta])
    # mcmc = pm.MCMC(model)
    # mcmc = pm.MCMC([observation, likelihood_knlogn, alpha, beta], db='pickle', dbname='results/runtime_model.pickle')
    map_ = pm.MAP(model)
    map_.fit()
    # mcmc = pm.MCMC([observation, likelihood_knlogn, alpha, beta])
    mcmc = pm.MCMC(model)
    mcmc.sample(iter=50000)
    # fig = plot(mcmc)
    # plot.savefig("testing")
    print("mcmc_model end")
    mcmc.db.close()
    return mcmc


#  predict values for y
def mcmc_predict(trace, x1_f, x2_f, equation):
    # standard_dev = 1
    #
    # alpha = np.array(trace.get_values('alpha'))
    # mu_alpha = np.average(alpha)
    # std_alpha = standard_dev * np.std(alpha)
    #
    # beta = np.array(trace.get_values('beta'))
    # mu_beta = np.average(beta)
    # std_beta = standard_dev * np.std(beta)
    #
    # ceta = np.array(trace.get_values('ceta'))
    # mu_ceta = np.average(ceta)
    # std_ceta = standard_dev * np.std(ceta)
    #
    # gamma = np.array(trace.get_values('gamma'))
    # mu_gamma = np.average(gamma)
    # std_gamma = standard_dev * np.std(gamma)
    #
    # params = (mu_alpha, mu_beta, mu_ceta, mu_gamma)
    # params_upper = (mu_alpha + std_alpha, mu_beta + std_beta, mu_ceta + std_ceta, mu_gamma + std_gamma)
    # params_lower = (mu_alpha - std_alpha, mu_beta - std_beta, mu_ceta - std_ceta, mu_gamma - std_gamma)
    #
    # y_pred = likelihood_knlogn(params, x1_f, x2_f, equation)
    # y_pred_upper = likelihood_knlogn(params_upper, x1_f, x2_f, equation)
    # y_pred_lower = likelihood_knlogn(params_lower, x1_f, x2_f, equation)
    #
    # # return y_pred, y_pred_upper, y_pred_lower
    return 0


def mcmc_fit(xdata, ytime, eq_used):

    print("length xdata:", len(xdata))
    print("length ytime:", len(ytime))

    for equation in eq_used:
        params = [xdata[:, 0], xdata[:, 1], ytime, equation]

        # creating mcmc model
        trace = mcmc_model(params)

        alpha_samples = trace.trace('alpha')[20000:]
        beta_samples = trace.trace('beta')[20000:]

        alpha_med = np.median(alpha_samples)
        beta_med = np.median(beta_samples)

        plot_samples(alpha_samples, beta_samples)
        simple_predict(trace)
        # save the learned model in pickle file
        # pickle.dump(trace, open("results/model" + str(equation) + ".pickle", "wb"), protocol=-1)

        # traceplot(trace)
        # move plot about learned param values to results folder
        # os.rename("mcmc.png", "results/mcmc_" + str(equation) + ".png")
        # plt.close()
    print("mcmc_fit finish")
    # [y_cal, y_cal_upper, y_cal_lower, size]

    return trace


def simple_predict(trace):
    x1_features, x2_size, y_runtime = load_test_data('covertype', 'runtimes/')

    alpha_samples = trace.trace('alpha')[20000:]
    beta_samples = trace.trace('beta')[20000:]

    alpha_med = np.median(alpha_samples)
    beta_med = np.median(beta_samples)

    y_pred = cal_likelihood(x2_size, x1_features, alpha_med, beta_med)
    y_predicted = list()
    y_predicted.append([y_pred, y_pred, y_pred, 1])

    plot_predictions(x1_features, x2_size, y_runtime, y_predicted, 'ct', 'ct')


def cal_likelihood(n, k, alpha, beta):
    n_log = np.log2(n)
    n_log_square = n_log ** 2

    val = alpha + beta + (k * n * n_log_square)

    return val

def plot_samples(alpha_samples, beta_samples):
    # figsize(12.5, 6)

    # histogram of the samples:
    plt.subplot(211)
    plt.title(r"Posterior distributions of the variables $\alpha, \beta$")
    plt.hist(beta_samples, histtype='stepfilled', bins=35, alpha=0.85,
         label=r"posterior of $\beta$", color="#7A68A6", normed=True)
    plt.legend()

    plt.subplot(212)
    plt.hist(alpha_samples, histtype='stepfilled', bins=35, alpha=0.85,
             label=r"posterior of $\alpha$", color="#A60628", normed=True)
    plt.legend()

    plt.savefig("results/params.png")
    plt.close()
