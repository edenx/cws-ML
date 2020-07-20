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

# g) Perform Kmeans on scaled data ----------------------------------------------------



if __name__=='__main__':
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
