import numpy as np
from matplotlib import pyplot as plt

# Question 1 -------------------------------------------------------------------------
n = 8
p = 2
K = 2

x1 = np.array([1, 1, 0, 0.5, 4, 6, 5, 5.5])
x2 = np.array([3, 3.5, 4, 4.2, 1, 0.5, 0, 1.2])
x = np.array([x1, x2])

# (a) Plot the Observation
plt.figure()
plt.scatter(x[0, :], x[1, :])
plt.title('Observations')
plt.xlabel("X1")
plt.ylabel("X2")
plt.savefig('fig00.png')
plt.show()
plt.close()

# (b) Initialise the Label
rs = np.random.RandomState(seed=123)
labels_new = rs.choice(K, size=n)

# report the initialised Label
plt.figure()
plt.scatter(x[0, :], x[1, :], c=labels_new)
plt.title('Initialised label with 2 clusters')
plt.xlabel("X1")
plt.ylabel("X2")
plt.savefig('fig01.png')
plt.show()
plt.close()

centroids = np.zeros((K, p))
flag = True
dist_ = np.zeros((K, n))

while flag:
     # update label
     labels = labels_new

     # (c) Compute the Centroids
     for i in range(K):
          centroids[i] = np.mean(x[:, labels==i], axis=1)

     # (d) Reassign Labels According to the Nearest Centroids
     for i in range(K):
          dist_[i] = np.sum((x-centroids[i][:, np.newaxis])**2, axis=0)

     labels_new = np.argmin(dist_, axis=0)

     # continue if label is changing
     flag = any((labels - labels_new) != 0)

     
# report the initialised Label
plt.figure()
plt.scatter(x[0, :], x[1, :], c=labels_new)
plt.title('Cluster label obtained from kmeans, k=2')
plt.xlabel("X1")
plt.ylabel("X2")
plt.savefig('fig02.png')
plt.show()
plt.close()
