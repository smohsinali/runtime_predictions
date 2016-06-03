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
import numpy as np
import matplotlib.pyplot as plt

# x_runtime = np.loadtxt("mnist.np")
x_runtime = np.genfromtxt("2016_runtime_data/results_3_bac.csv", skip_header=1, delimiter=',', dtype=None)
# y_runtime = np.loadtxt("runtimes/y_train_mnist.np")
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

































