import emcee
import corner
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator


def lnprior(theta):
    m, b, lnf = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf


def likelihood_knlogn(theta, x, y, equation, yerr):

    w, lnf = theta
    n, k = x

    if equation == 1:
        n_log = np.log(n) ** 2
        n_mod = n
        model = w[0] + w[1] * (k * n_mod * n_log)
        inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))

        return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

    return 0


def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + likelihood_knlogn(theta, x, y, yerr)


def ml(x, y):
    m_true = b_true = f_true = 1
    chi2 = lambda *args: -2 * likelihood_knlogn(*args)
    result = op.minimize(chi2, [m_true, b_true, np.log(f_true)], args=(x, y, yerr))
    m_ml, b_ml, lnf_ml = result["x"]
    print("""Maximum likelihood result:
        m = {0} (truth: {1})
        b = {2} (truth: {3})
        f = {4} (truth: {5})
    """.format(m_ml, m_true, b_ml, b_true, np.exp(lnf_ml), f_true))





