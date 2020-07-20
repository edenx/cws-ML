def ker_Gauss(X, Y, l):
     n = X.shape[0]
     m = Y.shape[0]

     # binomial formula ~ Smola's Blog
     tmp = np.tile(np.sum(X**2, axis=1, keepdims=True), m)
     tmp += np.tile(np.sum(Y**2, axis=1), (n,1))
     tmp -= 2 * X @ Y.T

     ker = np.exp(-tmp/(l**2))
     return ker

# generate samples around two concentric ellipse
def samp_ell(range_lis, n=40, seed=17671):

     import numpy.random as rm
     rs = rm.RandomState(seed=seed)
     lis_samp = []

     for i in range_lis:
          # generate random angle       
          phi = rs.uniform(0, 2*np.pi, n)
          # random rotation
          theta = rs.uniform(0, 2*np.pi, 1)

          samp = np.array([np.cos(phi) * i[0], np.sin(phi) * i[1]]).reshape((2, n))
          # rot_samp += [rot_mat @ samp + rs.normal(0, 0.3, size=(2, n))]
          lis_samp += [samp + rs.normal(0, 0.3, size=(2, n))]

     return lis_samp

def power_iter1(gram_mat, maxiter=700, seed=17671):
     
     # randomly generate a vector to initialise
     rs = rm.RandomState(seed=seed)
     v = rs.uniform(size=(gram_mat.shape[0], 1))
     l = 0

     # recursively compute the update
     n = 0
     while n < maxiter :
          n += 1

          z = gram_mat @ v
          v = z/LA.norm(z)
          l = v.T @ gram_mat @ v

     return l[0][0], v.flatten()

def power_iter(gram, k, maxiter=700, seed=17671):
     eval_lis = []
     evec_lis = []

     for i in range(k):
          l, v = power_iter1(gram, maxiter=maxiter, seed=17671)
          # print(v, '\n')
          eval_lis += [l]
          evec_lis += [v]

          # project on to subspace of v
          gram -= l * np.outer(v, v)
          
     # return the top k evecs and evals
     eval_arr = np.array(eval_lis)
     print('verify', eval_arr)
     evec_arr = np.array(evec_lis)/np.sqrt(eval_arr)[:, np.newaxis]

     return eval_arr, evec_arr

def centering(ker):
     n, m = ker.shape

     return ((np.eye(n) - np.ones((n,n))/n) @ ker @ (np.eye(m) - np.ones((m,m))/m))

def kPCA(dat, test, lab, kernel, hyper, k=2, plot=True):
     m = dat.shape[1]

     # ker_mat = ((np.eye(m) - np.ones((m,m))/m) @ kernel(dat.T, dat.T, hyper) 
     #                                           @ (np.eye(m) - np.ones((m,m))/m))
     ker_mat = centering(kernel(dat.T, dat.T, hyper))
     
     eval_k, evec_k = power_iter(ker_mat, k, maxiter=700, seed=17671)

     ker_test = centering(kernel(dat.T, test.T, hyper))
     kpc = evec_k @ ker_test 

     if plot:
          from sklearn.decomposition import PCA
          from sklearn.preprocessing import StandardScaler

          pca = PCA(n_components=2)
          X_std = StandardScaler().fit_transform(dat.T)
          X_pca = pca.fit_transform(X_std)

          plt.rcParams["figure.figsize"] = [10, 4]
          fig, axs = plt.subplots(2,2)
          fig.suptitle('KPCA with Gaussian kernel $\sigma$={}'.format(hyper))

          # visualise the clustering of the first two dimension
          plt.figure()
          axs[0,0].scatter(dat[0], dat[1], c=lab)
          axs[0,0].set_title('Original Data')

          axs[0,1].scatter(kpc[0], kpc[1], c=lab)
          axs[0,1].set_title('Projection to first 2 KPC')

          axs[1,0].scatter(dat[0], dat[1], c=lab)
          axs[1,0].set_title('Original Data')

          axs[1,1].scatter(X_pca[:, 0], X_pca[:, 1], c=lab)
          axs[1,1].set_title('Projection to first 2 PC')

          for ax in axs.flat[[0,2]]:
               ax.set(
                    xlabel='Feature 1', 
                    ylabel='Feature 2')
          axs.flat[1].set(
                    xlabel="first principal component",
                    ylabel="second principal component")
          axs.flat[3].set(
                    xlabel="first principal component",
                    ylabel="second principal component")

          # Hide  y ticks for right plots.
          for ax in axs.flat:
               ax.label_outer()

          fig.savefig('KPC{}.png'.format(hyper))
          # plt.show()
          plt.close()


     return eval_k, evec_k, kpc

def Wrapper_plot(dat, kernel, hyper, k=2):

     eval_2, evec_2 = kPCA(dat, kernel, hyper, k=2)
     proj_dat

if __name__=='__main__':
     import matplotlib.pyplot as plt
     import numpy as np
     import pandas as pd
     import time

     import numpy.random as rm
     from numpy import linalg as LA

     range_lis = [[4, 3.2], [1.2, 0.5]]
     n = 50

     samp = samp_ell(range_lis, n=n, seed=176)
     dat = np.concatenate((samp[0], samp[1]), axis=1)
     lab = np.repeat([0, 1], n)

     # l = 2.9
     # m = n * 2
     # # compute the centred Gram matrix
     # gram_mat = ((np.eye(m) - np.ones((m,m))/m) @ ker_Gauss(dat.T, dat.T, l) 
     #                                           @ (np.eye(m) - np.ones((m,m))/m))
     # # print(gram_mat)
     # lambda1, v1 = power_iter1(gram_mat)
     # # print('principle axis 1', lambda1, v1)
     # eval_k, evec_k = power_iter(gram_mat, k, maxiter=1000, seed=17671)

     # eval, evec = LA.eig(gram_mat)
     # # print(eval[:5])
     # # print(evec[:,0]-v1.flatten())

     # # compute the first k evals and evecs

     k = 2
     for hyper in np.linspace(0,3,num=30):     
          eval_k, evec_k, emb_k = kPCA(dat, dat, lab, ker_Gauss, hyper, k=2, plot=False)
     



