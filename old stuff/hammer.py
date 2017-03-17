import emcee
import corner
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as pl
from scipy import stats
from matplotlib.ticker import MaxNLocator


def lnprior(theta):
    m, b, lnf = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf


def likelihood_knlogn(w, x, y):

    n = x[:, 0]
    k = x[:, 1]

    equation = 1
    sd = w[2]

    if equation == 1:
        n_log = np.log(n) ** 2
        n_mod = n
        model = w[0] + w[1] * (k * n_mod * n_log)

        likelihood = -np.sum(stats.norm.logpdf(y, loc=model, scale=.0005))

        return likelihood

    return 0


def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + likelihood_knlogn(theta, x, y, yerr)


def hammer_fit(x, y, used_eq):
    w0 = w1 = sd = 1
    chi2 = lambda *args: likelihood_knlogn(*args)
    result = op.minimize(chi2, [w0, w1, sd], args=(x, y))
    w0, w1, sd = result["x"]
    print("""Maximum likelihood result:
        w0 = {0} (truth: n/a)
        w1 = {1} (truth: n/a)
        w2 = {2} (truth: n/a)
    """.format(w0, w1, sd))





