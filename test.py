# files = ["xdsf", "ysdfaf"]
# y = list(filter(lambda x : x.startswith("y"), files))
# print(y[0])
import numpy as np
from scipy.sparse import csr_matrix


A = csr_matrix(([1, 2, 0], ([0, 0, 3], [4, 0, 5]))).toarray()
# print(A)

data = open("datasets/mldata/dexter_train.data", "r")
lines = data.readlines()
# print(lines[0:2])
# data_array = np.zeros((len(lines), 20000))
test_array = np.zeros((2, 3))
x = ["0:20 1:41 2:32 \n", "0:2 1:3 2:4 \n"]
# x = x.split("\n")
print(len(lines))
data = np.array([0])
rows = np.array([0])
cols = np.array([0])

for i in range(len(lines)):
    xx = lines[i].split()
    for j in xx:
        vals = j.split(":")
        rows = np.insert(rows, 0, int(i))
        cols = np.insert(cols, 0, int(vals[0]))
        data = np.insert(data, 0, int(vals[1]))

# print(data, rows, cols)
c = csr_matrix((data, (rows, cols)))
a = c.toarray()
print(a.shape)
