# Question 2 ------------------------------------------------------------------------
# (a) Generate Data for Performing Classification Algorithms --------------------------

def genData(n, K, m=50, random_state=132):
     # Function to generate data 
     # n: number of observations in each class
     # K: number of classes
     # m: number of features

     import random
     import numpy as np
     import numpy.random as rm
     from sklearn.datasets import make_spd_matrix

     rs = rm.RandomState(seed=random_state)
     labels = np.repeat(np.arange(K), n)
     n_tot = n*K
     # assign features to distributions to be sampled
     findex = rs.choice(np.arange(1,6), m)

     # features from Gaussian Mixture with 3 Modes
     simDat = np.zeros((m,n_tot))
     for i in np.where(findex==1)[0]:
          simTri = rs.multivariate_normal(mean=rs.uniform(0,10,size=K),
               cov=make_spd_matrix(K, random_state=int(rs.choice(23722+i*11,1))), size=n)
          simDat[i] = simTri.T.flatten()

     # features from standard Cauchy
     simDat[np.where(findex==2)[0],:] = rs.standard_cauchy(size=(sum(findex==2),n_tot))
     
     # features from Gaussian(mu,sigma)
     simDat[np.where(findex==3)[0],:] = rs.normal(loc=rs.uniform(0,5,1),
                                        scale=rs.uniform(1),size=(sum(findex==3),n_tot))
     
     # features from Uniform(0,8)
     simDat[np.where(findex==4)[0],:] = rs.uniform(0,8,size=(sum(findex==4),n_tot))
     
     # features from Gamma(2,2)
     simDat[np.where(findex==5)[0],:] = rs.gamma(2,2,size=(sum(findex==5),n_tot))

     return {'Data':simDat, 'labels':labels}

# c, d, e) Perform Kmeans with K=2,3,4 ----------------------------------------------

def fit_kmeans(dat, labels, random_state=132):
     # Wrapper function to produce plot
     # dat: Data to be classified
     # labels: Original label associated with the data

     from matplotlib import pyplot as plt
     from sklearn.cluster import KMeans
     from sklearn.metrics import confusion_matrix

     conf_mat = dict()
     plt.rcParams["figure.figsize"] = [10, 8]
     fig, axs = plt.subplots(2, 2)

     # visualise the clustering of the first two dimension
     plt.figure()
     fig.suptitle('Kmeans on Untransformed Data')
     for K in range(2,5):
          kmeans = KMeans(n_clusters=K, random_state=random_state)
          labels_k = kmeans.fit_predict(dat)
          
          axs[K//2-1, K%2].scatter(dat[:,0], dat[:,1], c=labels_k)
          axs[K//2-1, K%2].set_title('Colored by clusters, k={}'.format(K))  
          
          # check the confusion matrix
          conf_mat['k={}'.format(K)] = confusion_matrix(labels, labels_k)
     
     axs[1, 1].scatter(dat[:,0], dat[:,1], c=labels)
     axs[1, 1].set_title('Colored by true classes')

     for ax in axs.flat:
          ax.set(xlabel='Feature 1', ylabel='Feature 2')

     # Hide x labels and tick labels for top plots and y ticks for right plots.
     for ax in axs.flat:
          ax.label_outer()

     fig.savefig("fig3.png")
     plt.show()
     plt.close()

     return conf_mat

# f) Perform Kmeans on PCA transformed data ------------------------------------------

def fit_kmeansT(K, dat, labels, random_state=132, plot=True, index=None):
     # Wrapper function to produce plot
     # K: number of clusters for Kmeans
     # dat: data to be classified
     # labels: original label associated with the data

     from matplotlib import pyplot as plt
     from sklearn.cluster import KMeans
     from sklearn.metrics import confusion_matrix

     kmeans3 = KMeans(n_clusters=K, random_state=random_state)
     labels_pc3 = kmeans3.fit_predict(dat)

     print(dat.shape)

     if plot:
          plt.rcParams["figure.figsize"] = [10, 4]
          fig, (axs1, axs2) = plt.subplots(1, 2)
          fig.suptitle('Kmeans on Transformed Data')

          # visualise the clustering of the first two dimension
          plt.figure()
          axs1.scatter(dat[:,0], dat[:,1], c=labels)
          axs1.set_title('Colored by true classes')

          axs2.scatter(dat[:,0], dat[:,1], c=labels_pc3)
          axs2.set_title('Colored by clusters of k-means, k=3')

          for ax in (axs1, axs2):
               ax.set(xlabel='Feature 1', 
                      ylabel='Feature 2')

          # Hide  y ticks for right plots.
          axs2.label_outer()

          fig.savefig('fig{}.png'.format(index))
          plt.show()
          plt.close()

     # check the confusion matrix
     return confusion_matrix(y, labels_pc3)

# Question 3-------------------------------------------------------------------------

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

     # Queastion 2 --------------------------------------------------------------------
     # use default random state for analysis, i.e. 132
     import numpy as np
     from matplotlib import pyplot as plt

     # output matrix to latex code
     import array_to_latex as a2l
     to_tex = lambda A : a2l.to_ltx(A, frmt = '{:6.2f}', 
                         arraytype = 'array', mathform=True)

     # a) Generate 25 obs in each of three classes with 50 features ------------------
     n=25; K=3; m=50
     sim = genData(n, K, m)
     simDat = sim['Data']
     labels = sim['labels']

     print(simDat.shape)
     # Visualise in first two features
     plt.figure()
     plt.scatter(simDat[0], simDat[1], c=labels)
     plt.xlabel('Feature 1')
     plt.ylabel('Feature 2')
     plt.title('Generated Data')
     plt.savefig('fig1.png')
     plt.show()
     plt.close()

     # b) Perform PCA ---------------------------------------------------------------
     from sklearn.decomposition import PCA
     from sklearn.preprocessing import StandardScaler
     X = simDat.T
     y = labels
     
     pca = PCA(n_components=3)
     X_std = StandardScaler().fit_transform(X)
     X_pca = pca.fit_transform(X_std)

     # visualise with first two components
     plt.figure()
     plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
     plt.xlabel('Principal Component 1')
     plt.ylabel('Principal Component 2')
     plt.title('PCA Transformed Data')
     plt.savefig('fig2.png')
     plt.show()
     plt.close()

     # c, d, e) Perform Kmeans with K=2,3,4 ------------------------------------------
     conf_mat = fit_kmeans(X, y)
     print('Confusion matrix for k=2', '\n', conf_mat['k=2'], '\n')
     to_tex(conf_mat['k=2'])
     print('\n')

     print('Confusion matrix for k=3', '\n', conf_mat['k=3'], '\n')
     to_tex(conf_mat['k=3'])
     print('\n')
     
     print('Confusion matrix for k=4', '\n', conf_mat['k=4'], '\n')
     to_tex(conf_mat['k=4'])
     print('\n')
     # all performs really badly

     # f) Perform Kmeans with K=3 on first 2 component PC of PCA ---------------------
     conf_matPCA = fit_kmeansT(3, X_pca[:,0:2], y, plot=True, index=4)
     print('Confusion matrix for k=3 with PC=1:2', '\n', conf_matPCA, '\n')
     to_tex(conf_matPCA)
     print('\n')
     # really good

     conf_matPCA_full = fit_kmeansT(3, X_pca, y, plot=True, index=5)
     print('Confusion matrix for k=3 with all of PC', '\n', conf_matPCA_full, '\n')
     to_tex(conf_matPCA_full)
     print('\n')
     # all perfectly classified

     # g) Perform Kmeans with K=3 on scaled data -------------------------------------
     from sklearn.preprocessing import scale
     X_scale = scale(X)

     conf_matScale = fit_kmeansT(3, X_scale, y, plot=True, index=6)
     print('Confusion matrix for k=3 with scaled data', '\n', conf_matScale, '\n')
     to_tex(conf_matScale)
     print('\n')
     # all perfectly classified, due to the 10 features generated from Cauchy

     # Question 3 ---------------------------------------------------------------------
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