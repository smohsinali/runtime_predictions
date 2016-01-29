files = ["xdsf", "ysdfaf"]
y = list(filter(lambda x : x.startswith("y"), files))
print(y[0])
x = [[1,2], [2,3]]
for i in x:
    print(i)
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