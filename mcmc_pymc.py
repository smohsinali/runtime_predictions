import pymc3 as pm
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import re


if __name__ == "__main__":
    a = "/home/alis/HPOlib/benchmarks/HPOlib-logreg/cv/irace_1_07_1000_2016-1-13--15-15-30-356070/Instances//22121"
    print(re.compile("//").split(a)[-1])
    # np.random.seed(123)
    # data = np.random.randn(20)
    #
    # with pm.Model():
    #     mu = pm.Normal('mu', 0, 1)
    #     sigma = 1.
    #     returns = pm.Normal('returns', mu=mu, sd=sigma, observed=data)
    #
    #     step = pm.Metropolis()
    #     trace = pm.sample(10000, step)
    #
    # sns.distplot(trace[2000:]['mu'], label='PyMC3 sampler')
    # plt.legend()
    # plt.savefig("mcmc_pymc.png")