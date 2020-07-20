
# Simulate Data
def create_features(t, phi, random_state=13251):
     import numpy as np
     import numpy.random as rm
     rs = rm.RandomState(seed=random_state)

     n = len(t)
     x = np.sin(t) * np.cos(phi) + rs.normal(0,0.5,n)
     y = np.sin(t) * np.sin(phi) + rs.normal(0,0.5,n)
     z = np.cos(t) + rs.normal(0,0.5,n)

     return np.array([x,y,z]).T

def simData(m=30, random_state=13251):
     import numpy as np
     import numpy.random as rm
     rs = rm.RandomState(seed=random_state)

     labels = np.repeat(np.array([0,1,2]), m)
     genDat = np.zeros((3*m,3))

     t1 = rs.uniform(-1,1,m) * np.pi/8
     t2 = rs.uniform(-1,1,m) * np.pi/4 + np.pi/2
     t3 = rs.uniform(-1,1,m) * np.pi/4 + np.pi/2
     t_lis = [t1, t2, t3]

     phi1 = rs.uniform(0,1,m) * np.pi*2
     phi2 = rs.uniform(-1,1,m) * np.pi/4
     phi3 = rs.uniform(-1,1,m) * np.pi/4 + np.pi/2
     phi_lis = [phi1, phi2, phi3]

     init = 0 
     for i in range(3):
          genDat[init:(init+m), :] = create_features(t_lis[i], phi_lis[i])

          init += m
     
     return {'Data':genDat, 'labels':labels}

# k-means
def kmeans(dat, K, random_state=13251, max=30, flag=True, plot=False):
     import numpy as np
     import numpy.random as rm
     from matplotlib import pyplot as plt

     n = dat.shape[0]
     p = dat.shape[1]
     x = dat.T

     # Initialise the Label; centroids; distance to centroids
     rs = rm.RandomState(seed=random_state)

     labels_new = rs.choice(K, size=n)
     centroids_new = np.zeros((K, p))
     dist_ = np.zeros((K, n))

     n = 0
     while flag:
          n += 1
          labels = labels_new
          centroids = centroids_new

          # update cluster centres
          for i in range(K):
               # if no assignment to class i, set the centre to infinity
               if x[:, labels==i].size==0:
                    centroids_new[i] = float("inf")
               # otherwise, update the centres accordingly
               else:
                    centroids_new[i] = np.mean(x[:, labels==i], axis=1)
          # find the distance to centres
          for i in range(K):
               dist_[i] = np.sum((x-centroids_new[i][:, np.newaxis])**2, axis=0)

          # find the sum of squared distances of samples to their closest cluster centre.
          inertia_ = np.sum(np.amin(dist_, axis=0))
          labels_new = np.argmin(dist_, axis=0)

          # continue if label is changing
          flag = any((labels - labels_new) != 0) and n<max

          if plot:
               plt.scatter(x[0, :], x[1, :], c=labels_new)
               plt.xlabel("x")
               plt.ylabel("y")
               plt.show()

     return {'labels':labels_new, 'inertia_':inertia_}



if __name__=='__main__':
     import numpy as np
     from matplotlib import pyplot as plt
     import numpy.random as rm

     # Simulate Data ------------------------------------------------------------------
     sim = simData(30)
     genDat = sim['Data']
     labels = sim['labels']

     # 3d plot
     from mpl_toolkits import mplot3d
     from mpl_toolkits.mplot3d import Axes3D

     fig = plt.figure()
     ax = fig.add_subplot(111, projection='3d')
     ax.scatter(genDat[:,0], genDat[:,1], genDat[:,2], c=labels, cmap='viridis', linewidth=0.5);
     plt.title('Simulated data')
     plt.savefig('fig7.png')
     plt.show()
     plt.close()

     # run with 3 clusters -----------------------------------------------------------
     from sklearn.metrics import confusion_matrix
     kmeans_clust3 = kmeans(genDat, 3, plot=False)
     pred_labs = kmeans_clust3['labels']
     print(confusion_matrix(pred_labs, labels))

     fig = plt.figure()
     ax = fig.add_subplot(111, projection='3d')
     ax.scatter(genDat[:,0], genDat[:,1], genDat[:,2], c=pred_labs, cmap='viridis', linewidth=0.5);
     plt.title('Colored by clusters of k-means, k=3')
     plt.savefig('fig8.png')
     plt.show()
     plt.close()

     # vary the number of clusters ---------------------------------------------------
     # plot the sum of square distances (objective) versus  k 
     # and then find the 'elbow'.

     krange = range(2, 10)
     obj = np.zeros(len(krange))
     for k in krange:
          clust = kmeans(genDat, k, plot=False)
          obj[k-2] = clust['inertia_']
     plt.figure()
     plt.plot(krange, obj)
     plt.title('Elbow plot with varying k for Kmeans')
     plt.xlabel('number of clusters specified, k')
     plt.ylabel('Sum of distance to closest cluster')
     plt.savefig('fig9.png')
     plt.show()
     plt.close()
     # the 'elbow' is roughly at 3

     # the kmeans function from scikit learn
     from sklearn.cluster import KMeans
     krange = range(2, 10)
     obj = np.zeros(len(krange))
     for k in krange:
          kmeans = KMeans(n_clusters=k, random_state=13251).fit(genDat)
          obj[k-2] = kmeans.inertia_
     plt.figure()
     plt.plot(krange, obj)
     plt.show()
     plt.close()