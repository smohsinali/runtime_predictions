
import numpy as np
import os
from shutil import copy2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# from sklearn import neighbors, datasets
from sklearn.cluster import KMeans

algo = 'dt'
# algo = 'rf'
# algo = 'sgd'

directory = 'runtimes/all_' + algo
dir_csv = 'datasets/allData'
# directory = 'runtimes/all_dt'
# directory = 'runtimes/all_dt'

data = np.genfromtxt("runtimes/" + algo + "_graphs_summary.csv", skip_header=1, delimiter=',', dtype=None)
ds_names = []
X = []
y = []
for i in data:
    ds_names.append(str(i[0], 'utf-8'))
    X.append([i[1], i[2], i[3]])
    y.append(i[3])
# X = [data[:,1], data[:,2]]
# Y = data[:,3]
X = np.array(X)
y = np.array(y)
# print(y)

n_neighbors = 5

clf = KMeans(init='k-means++', n_clusters=5, n_init=2)
# clf.fit(X)
clf.fit(X)
Z = clf.predict(X)
print(Z)

results = {}
for i, j in enumerate(Z):
    if j not in results:
        results[j] = [ds_names[i]]
    else:
        results[j].append(ds_names[i])

print(results)

for i in results:
    print(str(i) + ":" + str(len(results[i])))
    path = os.path.join(dir_csv, str(i))
    if not os.path.exists(path):
        os.makedirs(path)

    for name in results[i]:
        fnameX = os.path.join(dir_csv, 'results_' + name[0:-8] + '_bac.csv')
        # fnameY = os.path.join(dir_csv, 'res' + name + '.np')
        copy2(fnameX, path)
        # copy2(fnameY, path)
        print(fnameX)

