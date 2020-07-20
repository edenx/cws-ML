# Q1 Nearest Mean clustering  -----------------------------------------------------------

# Image pre-processing
def Wrapper_Dat(cla, dat):
     """Wrapper function to process image data from lfw into patched format.
     If teo classes are inputed, relabel the data into class +/-1
     -----------
     Input
     cla: list; containing a pair or one label of class in lfw 
     dat: 3d array; containing images in row 
     p: integer; dimension of the patch
     -----------
     Output
     processed images a/na relabeled classes
     """
     target = dat.target

     cla_index = np.concatenate([np.where(target == i)[0] for i in cla])
     arr_dat = dat.images[cla_index]

     # scale the original image
     arr_dat = ((arr_dat - np.mean(arr_dat, axis=(-1,-2), keepdims=True))
                         /np.std(arr_dat, axis=(-1,-2), keepdims=True))
     proc_imag = Wrapper_extra_patches3(arr_dat)

     # change the label to +/-1; if input class is 2-dim
     if len(cla) is 2:

          labels = target[cla_index]
          labels = [-1 if i==cla[0] else 1 for i in labels]

          return proc_imag, np.array(labels)

     elif len(cla) is 1:

          return proc_imag
     else:

          return proc_imag, np.array(target[cla_index])

def extra_patches3(imag):
     """Extracts 3 by 3 patches of any nd array in place using strides, 
     with stepsize 1
     -----------
     Input
     imag: np array storing images
     -----------
     Returns
      patches : strided 4d array
      4d array indexing patches in row and column order in the first two axes
      and containing patches on the last 2 axes. 
     """
     import numpy as np
     from numpy.lib.stride_tricks import as_strided

     # pad the image with np.nan, width=1
     imag = np.pad(imag, 1, mode='constant', constant_values=np.nan)

     # dimension of image and patches
     i_h, i_w = imag.shape
     p = 3
     n_patches = (i_h-p+1) * (i_w-p+1)

     imag_ndim = imag.ndim
     patch_shape = tuple([p] * imag_ndim)
     patch_strides = imag.strides

     # the shape of produced array of patches
     patch_indices_shape = np.array(imag.shape) - np.array(patch_shape) + 1
     # output of strided array of patches
     shape_out = tuple(list(patch_indices_shape) + list(patch_shape))
     strides_out = tuple(list(patch_strides) * 2) 
     patches = as_strided(imag, shape=shape_out, strides=strides_out)

     # renormalise by deviding the norm
     return patches/np.sqrt(np.sum(patches**2, axis=(-1,-2), keepdims=True)) # 4d array


def Wrapper_extra_patches3(arr_imag):
     """Wrapper function of extra_patches3 for each image in the data set
     """
     from itertools import starmap, product

     return np.array(list(starmap(extra_patches3, product(arr_imag))))

# kernel function 
def ker_Imag(X, Y, kernel, hyper, sigma):
     """Compute the distance between patches in one image with the other only if
     the overlap between two patches are above 50%
     -----------
     Input
     X, Y: 4d array; storing patched image  rocessed by 'extra_patches3'
     kernel: fuction; kernel function for patches
     hyper: scalar; hyperparameter of the kernel function
     sigma: sclar; hyper parameter of the Gaussian kernel of distances between patches
     -----------
     Return:
     scalar; mean of kernel evaluation between patches
     """
     # X, Y 4d array of same dimension, n, m, p, p
     n, m = X.shape[:2]
     X_true = X[1:(n-1), 1:(m-1), :, :]

     # distance^2 between >100%, <50% overlapped patch
     ker = [kernel(X_true, Y[1:(n-1), 1:(m-1), :, :], hyper)]
     # take centroids as the reference position for the patches
     for i in range(-1, 2, 2):
          # for the nearest overlaps, can find directly the distance between centroids of patches
          ker += [kernel(X_true, Y[(1+i):(n-1+i), 1:(m-1), :, :], hyper) * np.exp(-1/sigma**2)]
          ker += [kernel(X_true, Y[1:(n-1), (1+i):(m-1+i), :, :], hyper) * np.exp(-1/sigma**2)]
          ker += [kernel(X_true, Y[(1+i):(n-1+i), (1+i):(m-1+i), :, :], hyper) * np.exp(-2/sigma**2)]
          ker += [kernel(X_true, Y[(1-i):(n-1-i), (1+i):(m-1+i), :, :], hyper) * np.exp(-2/sigma**2)]

     return np.nanmean(np.array(ker))

def ker_Gauss3(u, v, l):
     """Gaussian kernel for vectors
     ----------
     Input
     u, v: 4d array; each position of first two axis stores patches as from 'extra_patches3'
     l: positive scalar; hyperparameter sigma 
     ----------
     Output
     kernel evaluation between u and v
     """
     # u, v 3 by 3 array
     return np.exp(-np.sum((u - v)**2, axis=(-1,-2))/l**2)

def Wrapper_ker_Gauss(X_arr, Y_arr, l, sigma):
     """Wrapper function for evaluating the between image kernel value with 'ker_Imag'
     for Guassian kernel
     ----------
     Input
     X_arr, Y_arr: 5d array; first axis stores patched images in class X and Y respectively
     l: positive scalar; hyperparameter of Gaussian kernel
     -----------
     Output
     2d array; storing between image kernel evaluations from two classes X and Y.
     """
     from itertools import starmap, product

     ker = np.array(list(starmap(ker_Imag, product(X_arr, Y_arr, [ker_Gauss3], 
                                                       np.array([l]), np.array([sigma])))))

     return ker.reshape(X_arr.shape[0], Y_arr.shape[0])

# nearest mean algorithm
def train_test_split(dat, lab, test_ratio=0.25, seed=17671):
     """Split dataset into train and test set (stratified)
     ----------------
     Input:
     dat: n darray; X
     lab: 1 darray; y
     test_ratio: [0,1]; test to train ratio
     ----------------
     Output:
     dat_train, dat_test, lab_train, lab_test
     """
     import numpy.random as rm
     rs = rm.RandomState(seed=seed)

     ind_test_lis = []; ind_train_lis = []

     for i in np.unique(lab):
          index = np.where(lab==i)[0]
          ind_test = rs.choice(index, round(len(index) * test_ratio), replace=False)

          ind_test_lis += list(ind_test)
          ind_train_lis += list(index[~np.isin(index, ind_test)])

     
     dat_test = dat[ind_test_lis]; dat_train = dat[ind_train_lis]
     lab_test = lab[ind_test_lis]; lab_train = lab[ind_train_lis]

     return dat_train, dat_test, lab_train, lab_test
     

def NNmean(dat_train, dat_test, lab_train, lab_test, kernel, hyper, sigma):
     """Find the nearest mean to the training set
     """
     n_cla = np.unique(lab_test)
     dist_lis = []
     # compute the corresponding distance of test set to class mean of train set
     for i in n_cla:

          index_train = np.where(lab_train==i)[0]
          samp_train = dat_train[index_train]

          ker_test = kernel(dat_test, dat_test, hyper, sigma)

          dist_lis += [np.diag(ker_test)
                         - 2 * np.mean(kernel(dat_test, samp_train, hyper, sigma), axis=1) 
                         + np.mean(kernel(samp_train, samp_train, hyper, sigma))]
     
     dist_arr = np.array(dist_lis).reshape((len(n_cla), len(lab_test)))
     lab_pred = np.choose(np.argmin(dist_arr, axis=0), n_cla)

     # confusion matrix
     conf_mat = pd.crosstab(pd.Series(lab_test, name="actual"), 
                              pd.Series(lab_pred.astype('int'), name="predict"))
     
     # print('confusion matrix', conf_mat)
     misclass = 1 - np.diag(conf_mat.values).sum()/conf_mat.values.sum()

     return {'lab_pred': lab_pred, 'conf_mat': conf_mat, 'error': misclass}


# visualisations
def Wrapper_Plot(kernel_dic, hyper_dic, arr_train, arr_test, lab_train, lab_test, p=3):
     """Wrapper function to plot the missclassification rate with varying hyper 
     parameters.
     -------------
     Input:
     kernel_dic: dictionary; storing kernel name and corresponding function
     hyper_dic: dictionary; storing hyper parameter name and corresponding list 
     of values
     -------------
     Output: 
     Plot of misclassification rate against hyper parameter values
     """
     ker_name, kernel = list(kernel_dic.items())[0]

     # for j in range(len(hyper_dic)):
     hyper_name, hyper_lis = list(hyper_dic.items())[0]
     sigma_name, sigma_lis = list(hyper_dic.items())[1]
     # error_lis = []

     # for i in hyper_lis:
     #      pred_class = NNmean(arr_train, arr_test, lab_train, lab_test, kernel, i, 0.2)
     #      error_lis += [pred_class['error']]

     # min_error = min(error_lis)
     # argmin = hyper_lis[error_lis.index(min_error)]

     # log_hyper_lis = np.log10(hyper_lis)
     # plt.figure()
     # plt.plot(log_hyper_lis, error_lis, label='Error')
     # plt.hlines(min_error, xmin=log_hyper_lis.min(), xmax=log_hyper_lis.max(), 
     #           label='Min Error {}'.format(round(min_error,3)))
     # plt.title('Nearest Mean Clustering with {a} kernel/{b}=0.2'.format(a=ker_name, b=sigma_name))
     # plt.xlabel('$\log10 sigma_p^2$/ argmin $sigma_p^2$={}'.format(round(argmin,3)))
     # plt.ylabel('Error of Misclassification/p={}'.format(p))
     # plt.legend()
     # plt.savefig('{a}_{b}.png'.format(a=hyper_name, b=p))
     # plt.show()
     # plt.close()

     # error_lis = []
     # for i in sigma_lis:
     #      pred_class = NNmean(arr_train, arr_test, lab_train, lab_test, kernel, 0.1, i)
     #      error_lis += [pred_class['error']]

     # min_error = min(error_lis)
     # argmin = sigma_lis[error_lis.index(min_error)]

     # log_sigma_lis = np.log10(sigma_lis)
     # plt.figure()
     # plt.plot(log_sigma_lis, error_lis, label='Error')
     # plt.hlines(min_error, xmin=log_sigma_lis.min(), xmax=log_sigma_lis.max(), 
     #           label='Min Error {}'.format(round(min_error,3)))
     # plt.title('Nearest Mean Clustering with {a} kernel/{b}=0.1'.format(a=ker_name, b=hyper_name))
     # plt.xlabel('$\log10 sigma_l^2$/ argmin $sigma_l^2$={}'.format(round(argmin,3)))
     # plt.ylabel('Error of Misclassification/p={}'.format(p))
     # plt.legend()
     # plt.savefig('{a}_{b}.png'.format(a=sigma_name, b=p))
     # plt.show()
     # plt.close()   

     error_lis = []

     for i in sigma_lis:
          error_hyper = []

          for j in hyper_lis:
               pred_class = NNmean(arr_train, arr_test, lab_train, lab_test, kernel, j, i)
               error_hyper += [pred_class['error']]

          error_lis.append(error_hyper)

     log_hyper_lis = np.log10(hyper_lis)
     log_sigma_lis = np.log10(hyper_lis)

     plt.figure()
     plt.contour(log_hyper_lis, log_sigma_lis, np.array(error_lis))
     plt.title('Nearest Mean Clustering with {a} kernel'.format(a=ker_name))
     plt.xlabel('$\log_{c}$ {a}'.format(c=10, a=hyper_name))
     plt.ylabel('$\log_{c}$ {a}'.format(c=10, a=sigma_name))
     plt.legend()
     plt.savefig('contour_error.png')
     plt.show()
     plt.close()

     import seaborn as sns

     plt.figure()
     ax = sns.heatmap(np.array(error_lis)
                              , cbar_kws={'label': 'misclassification error'}
                              , cmap='YlGnBu'
                              , xticklabels=log_hyper_lis
                              , yticklabels=log_sigma_lis)
     ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
     ax.set_yticklabels(ax.get_yticklabels(), rotation=45, fontsize=8)
     ax.set_ylabel('$\log_{c}$ {a}'.format(c=10, a=sigma_name))
     ax.set_xlabel('$\log_{c}$ {a}'.format(c=10, a=hyper_name))
     ax.set_title('Nearest Mean Clustering with {a} kernel'.format(a=ker_name))
     plt.savefig('heat_error.png')
     plt.show()
     plt.close()


if __name__=='__main__':
     import matplotlib.pyplot as plt
     import numpy as np
     import pandas as pd
     import time

     # can we use this function, but what should be the `resize` argument? By default?
     from sklearn.datasets import fetch_lfw_people
     # Download the data, if not already on disk and load it as numpy arrays
     lfw_people = fetch_lfw_people(min_faces_per_person=40, resize=0.3)
     target = lfw_people.target
     # print('target has classes', np.unique(target, return_counts=True))

     # preprocessing -------------------------------------------------------------------------------
     # classes 1, 6, 10, 13, 14
     # with # images 42, 44, 42, 41, 41
     # cla=[1, 6, 10, 13, 14]; test_ratio = 0.25

     # classes 3, 5, 17
     # with # images 121, 109, 144
     cla=[3, 5, 17]; test_ratio = 0.25
     
     proc_imag, labels = Wrapper_Dat(cla, lfw_people)
     # print("dim of classes", np.unique(labels, return_counts=True))
     dat_train, dat_test, lab_train, lab_test = train_test_split(proc_imag, 
                                                  labels, test_ratio, seed=21203)
     print("dim of train ", dat_train.shape, '\n')
     print("dim of test ", dat_test.shape, '\n')
     print("train to test ratio ", np.unique(lab_train, return_counts=True), 
                                   np.unique(lab_test, return_counts=True))

     # # Varying hyperparams --------------------------------------------------------------------------

     kernel_dic = {
          'Gaussian': Wrapper_ker_Gauss,
          # 'VovkPolynomial': Wrapper_ker_Poly
          }
     hyper_dic = {
          '$\sigma_p$': np.logspace(-2,2,10),
          '$\sigma_l$': np.logspace(-2,2,10)
          }

     # a) Nearest Mean algorithm
     Wrapper_Plot(kernel_dic, hyper_dic, dat_train, dat_test, lab_train, lab_test)




