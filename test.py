# files = ["xdsf", "ysdfaf"]
# for i in range(3):
#     for t in range(5):
#         for k in range(2):
#             print(5 * 2 * i + (2 * t + k))
# y = list(filter(lambda x : x.startswith("y"), files))
# print(y[0])
# x = [[1,2], [2,3]]
# for i in x:
#     print(i)
# import numpy as np
# from scipy.sparse import csr_matrix
#
#
# def load_dexter(data, labels):
#     lines = data.readlines()
#
#     values = np.array([0])
#     rows = np.array([0])
#     cols = np.array([0])
#
#     for i in range(len(lines)):
#         line_list = lines[i].split()
#         for j in line_list:
#             vals = j.split(":")
#             rows = np.insert(rows, 0, int(i))
#             cols = np.insert(cols, 0, int(vals[0]))
#             values = np.insert(values, 0, int(vals[1]))
#
#     data = csr_matrix((data, (rows, cols)))
#     labels = np.loadtxt(labels)
#
#     return data, labels
#
#
# if __name__ == "__main__":
#     data = open("datasets/mldata/dexter_train.data", "r")
#     labels = open("datasets/mldata/dexter_train.labels", "r")
#     load_dexter(data, labels)
# import numpy as np
# import matplotlib.pyplot as plt
#
# # x_runtime = np.loadtxt("mnist.np")
# x_runtime = np.genfromtxt("2016_runtime_data/results_3_bac.csv", skip_header=1, delimiter=',', dtype=None)
# # y_runtime = np.loadtxt("runtimes/y_train_mnist.np")
print("hello")

# x = x_runtime[0:90]
# arr = range(900)
# y = [sum(y_runtime[i:900:90])/10.0 for i in range(90)]
# x2 = [sum(x_runtime[i:900:90, 2])/10.0 for i in range(90)]
# x1 = x_runtime[0:90, 0]
# print()
#
# plt.plot(x1, x2, label="Dims 6000X5000")
# plt.xlabel("Number of Instances")
# plt.ylabel("Depth of Tree")
# plt.title("Dataset:Gisette")
# plt.legend(loc='best')
# # plt.savefig("results/mc_" + data_name + "_pred_highres.png", dpi=300)
# plt.savefig("depth_Gisette.png", dpi=200)
# # np.savetxt("x_gisette10.np", x)
# # np.savetxt("results/y_gisette10.np", y)
# #
#
#
# from tabulate import tabulate
#
# print(tabulate({"Name": ["Alice", "Bob"], "xcd": tabulate({"Age": [24, 19], "foo": [3, 4]}, headers="keys")}, headers="keys"))

# a = 2
# b = 3
#
#
#
# class SubValueError(ValueError):
#     def __init__(self, errorArgs, a, b):
#         ValueError.__init__(self, "custom value error")
#         self.eror = errorArgs
#         self.a = a
#         self.b = b
#
# class SubValueError2(ValueError):
#     pass
#
# try:
#     raise SubValueError2('HiThere')
# except:
#     raise
# except SubValueError as sve:
#     print('An exception flew by!')
#     print("sve", sve.eror)
#     print("a", sve.a)
#     raise
#
# except SubValueError2 as sve:
#     print('An exception flew by!')
#     raise

#
# def rec(a):
#     if len(a) == 0:
#         return
#     rec(a[1:])
#     return(a[0])
#
#
#
# a = rec("cdf")
# print(a)

import numpy as np

# a = np.array([
#             [2, 4, 5],
#             [3, 5, 3]
#             ])
# np.savetxt("test.csv", a, delimiter=",", fmt=["%d", "%d", "%0.6f"], header="fkd, djls, jdls")


from sklearn.ensemble import RandomForestRegressor
from pymc3 import Model, Normal, HalfNormal, Deterministic
from pymc3 import find_MAP
from pymc3 import NUTS, sample, Metropolis
from pymc3 import traceplot
from scipy import optimize
import matplotlib.pyplot as plt

# ans = lambda x: 2*x + 230
# answer = lambda a,b,x : a*x + b

# def answer(a,b,x):
#     val = a*x + b
#     return val

# x_test = 270

# x = np.array([i for i in range (100)])
# y = np.array([ans(j) for j in x])

# x_train = x[0:1000:100]
# y_train = y[0:1000:100]
#
# reg = RandomForestRegressor(n_estimators=100)
# reg.fit(x_train,y_train)
# result = reg.predict(x_test)
#
# print(ans(x_test), result)
# print(x)
# # print(y)
# N=1000
# a,b = 2.0, 2300.0
# np.random.seed(47)
# # X = np.linspace(0, 100, N)
# X = np.arange(N)
# Y = a * X + b
# basic_model = Model()
# print(X,Y)
# with basic_model:
#
#     # ## Priors for unknown model params
#     alpha = Normal('alpha', mu=0, sd=1)
#     beta = Normal('beta', mu=10, sd=1)
#     sigma = HalfNormal('sigma', sd=1)
#     mu = alpha * X + beta
#
#     # ## Likelihood of obs
#     y_obs = Normal('Y_obs', mu=mu, sd=sigma, observed=Y)
#     # start = find_MAP()
#     # step = Metropolis(scaling=start)
#     # trace = sample(500, step, start=start)
#     # ## obtain starting values via MAP
#     start = find_MAP(fmin=optimize.fmin_powell)

    # ## draw posterior samples

    # using NUTS
    # trace = sample(500, start=start)
    # Inference!
    # start = find_MAP()  # Find starting value by optimization
    # step = NUTS(scaling=start)  # Instantiate MCMC sampling algorithm
    # trace = sample(2000, step, start=start)  # draw 2000 posterior samples using NUTS sampling


#     # using metropolis hastings with 2000 burin steps
#     step = Metropolis([alpha, beta])
#     sample(15000, start=start, step=step)
#     trace = sample(50000,start=start, step=step)
#
# mu_alpha = np.average(np.array(trace.get_values('alpha')))
# mu_beta = np.average(np.array(trace.get_values('beta')))
#
# print("\nalpha", mu_alpha)
# print("beta", mu_beta)
# predicted = [(mu_alpha*i+mu_beta) for i in X]
# print(predicted)
# plt.plot(X, Y)
# plt.plot(X, predicted)
#
# # plt.show()
#
# ###################################################
# ax, plt = traceplot(trace)
# plt.show()
# plt.savefig("what.png", dpi=200)


# a = input("type something:")
# print("you typed:" + a)










