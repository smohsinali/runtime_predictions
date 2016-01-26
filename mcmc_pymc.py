import numpy as np
import matplotlib.pyplot as plt
from pymc3 import Model, Normal, HalfNormal
from pymc3 import find_MAP
from pymc3 import NUTS, sample
from scipy import optimize
from pymc3 import traceplot
from pymc3 import summary


def mcmc_fit(x_runtime, y_runtime):
    # true param values
    alpha, sigma = 1, 1
    beta = 3

    # variables
    x1 = x_runtime[:30, 1]
    x2 = x_runtime[:30, 0]
    y = y_runtime[:30]
    # print(y_runtime, x1, x2)
    # Y = y_runtime_train

    basic_model = Model()

    with basic_model:

        # Priors for unknown model params
        alpha = Normal('alpha', mu=0, sd=10)
        beta = Normal('beta', mu=0, sd=10, shape=1)
        sigma = HalfNormal('sigma', sd=1)

        # Expected value of outcome
        mu = alpha + beta * (x1 * x2 ** 2 * np.log2(x2))

        # Likelihood of obs
        y_obs = Normal('Y_obs', mu=mu, sd=sigma, observed=y)

        # obtain starting values via MAP
        start = find_MAP(fmin=optimize.fmin_powell)

        # draw 2000 posterior samples
        trace = sample(50, start=start)

    print(start)
    print(trace["alpha"])
    # traceplot(trace)
    summary(trace)
    vars = trace.varnames
    var = None

    alpha = np.array(trace.get_values('alpha'))
    mu_alpha = np.average(alpha)
    std_alpha = 1 * np.std(alpha)

    beta = np.array(trace.get_values('beta'))
    mu_beta = np.average(beta)
    std_beta = 1 * np.std(beta)

    y_cal = mu_alpha + mu_beta * (x_runtime[:, 1] * x_runtime[:, 0] ** 2 * np.log2(x_runtime[:, 0]))
    y_cal_upper = (mu_alpha + std_alpha) + (mu_beta + std_beta) * (x_runtime[:, 1] * x_runtime[:, 0] ** 2 * np.log2(x_runtime[:, 0]))
    y_cal_lower = (mu_alpha - std_alpha) + (mu_beta - std_beta) * (x_runtime[:, 1] * x_runtime[:, 0] ** 2 * np.log2(x_runtime[:, 0]))

    sigma = np.array(trace.get_values('sigma'))

    plt.plot(x_runtime[:, 0], y_runtime)
    plt.plot(x_runtime[:, 0], y_cal)
    plt.xlabel("Number of Instances")
    plt.ylabel("Time(s)")
    plt.title("Dataset: adult\nPrediction ")
    plt.fill_between(x_runtime[:, 0], y_cal_upper, y_cal_lower, facecolor='yellow', alpha=0.2)
    plt.savefig("mc_adult_pred.png", dpi=300)

    # for var in vars:
    #     # Extract sampled values
    #     print(var)
    #     samples = np.array(trace.get_values(var))
    #     print(samples)

    print("\nmcmc_fit end")

if __name__ == "__main__":

    x_runtime = np.loadtxt("x_runtime_train_a.np")
    y_runtime = np.loadtxt("y_runtime_train_a.np")
    mcmc_fit(x_runtime, y_runtime)
    print("plot saved in mcmc.png")

    # percent_data = 10
    # train_size = x_runtime.shape[0]/percent_data
    # test_size = x_runtime.shape[0]/(100 - percent_data)
    #
    # print(type(x_runtime))
    # print(x_runtime.shape)
    # print(train_size)

    # x_train, x_test, y_train, y_test = train_test_split(x_runtime_train, y_runtime_train, train_size=.1)
    # x_runtime_train = x_runtime[0:train_size]
    # y_runtime_train = y_runtime[0:train_size]
    #
    # x_runtime_test = x_runtime[train_size + 1:]
    # y_runtime_test = y_runtime[train_size + 1:]












