
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
def moving_window3(X, Y, kernel, hyper):
     """Compute the distance between patches in one image with the other only if
     the overlap between two patches are above 50%
     -----------
     Input
     X, Y: 4d array; storing patched image processed by 'extra_patches3'
     kernel: fuction; kernel function for patches
     hyper: scalar; hyperparameter of the kernel function
     -----------
     Return:
     scalar; mean of kernel evaluation between patches
     """
     # X, Y 4d array of same dimension, n, m, p, p
     n, m = X.shape[:2]
     X_true = X[1:(n-1), 1:(m-1), :, :]

     # distance^2 between >100%, <50% overlapped patch
     ker = [kernel(X_true, Y[1:(n-1), 1:(m-1), :, :], hyper)]
     for i in range(-1, 2, 2):
          ker += [kernel(X_true, Y[(1+i):(n-1+i), 1:(m-1), :, :], hyper)]
          ker += [kernel(X_true, Y[1:(n-1), (1+i):(m-1+i), :, :], hyper)]

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

def Wrapper_ker_Gauss(X_arr, Y_arr, l):
     """Wrapper function for evaluating the between image kernel value with 'moving_window3'
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

     ker = np.array(list(starmap(moving_window3, product(X_arr, Y_arr, [ker_Gauss3], np.array([l])))))

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
     # split into -/+1 classes
     ind = lab<0
     neg_dat = dat[ind]; neg_lab = lab[ind]
     pos_dat = dat[~ind]; pos_lab = lab[~ind]

     # select test set in two classes respectively
     ind_neg = np.arange(neg_dat.shape[0])
     ind_pos = np.arange(pos_dat.shape[0])

     ind_test_neg = rs.choice(ind_neg, round(neg_dat.shape[0]*test_ratio), replace=False)
     ind_train_neg = np.delete(ind_neg, ind_test_neg)

     ind_test_pos = rs.choice(ind_pos, round(pos_dat.shape[0]*test_ratio), replace=False)
     ind_train_pos = np.delete(ind_pos, ind_test_pos)

     # split into test/train classes
     dat_test = np.concatenate((neg_dat[ind_test_neg], pos_dat[ind_test_pos]))
     dat_train = np.concatenate((neg_dat[ind_train_neg], pos_dat[ind_train_pos]))

     lab_test = np.concatenate((neg_lab[ind_test_neg], pos_lab[ind_test_pos]))
     lab_train = np.concatenate((neg_lab[ind_train_neg], pos_lab[ind_train_pos]))

     return dat_train, dat_test, lab_train, lab_test


def Varf(dat_train, dat_test, lab_train, lab_test, kernel, hyper=0.2):
     n_train = len(lab_train)
     n1_train = sum(lab_train==1); n2_train = sum(lab_train==-1)
     n1_test = sum(lab_test==1); n2_test = sum(lab_test==-1)

     # construct mu1-mu2 for training samples
     alpha = np.ones(n_train)
     alpha[lab_train==1] = 1/n1_train; alpha[lab_train==-1] = -1/n2_train

     # construct kernel matrix for the variance
     ker_train = kernel(dat_train, dat_train, hyper)
     ker_test = kernel(dat_test, dat_train, hyper)

     # norm of the difference of mean
     fnorm = alpha @ ker_train @ alpha[:, np.newaxis]
     print(alpha, '\n')
     print(ker_train)

     # Ka
     ka_train = ker_train @ alpha[:, np.newaxis]
     ka_test = ker_test @ alpha[:, np.newaxis]
     print('dim of ka_train', ka_train.shape, '\n')
     print('dim of ka_test', ka_test.shape, '\n')

     # train variance
     var1_train = (np.sum(ka_train[lab_train==1, :]**2) 
                    - 1/n1_train * np.sum(ka_train[lab_train==1, :])**2) / ((n1_train-1)*fnorm)
     var2_train = (np.sum(ka_train[lab_train==-1, :]**2) 
                    - 1/n2_train * np.sum(ka_train[lab_train==-1, :])**2) / ((n2_train-1)*fnorm)

     # test variance
     var1_test = (np.sum(ka_test[lab_test==1, :]**2) 
                    - 1/n1_test * np.sum(ka_test[lab_test==1, :])**2) / ((n1_test-1)*fnorm)
     var2_test = (np.sum(ka_test[lab_test==-1, :]**2) 
                    - 1/n2_test * np.sum(ka_test[lab_test==-1, :])**2) / ((n2_test-1)*fnorm)

     return var1_train, var2_train, var1_test, var2_test


if __name__=='__main__':
     # import matplotlib.pyplot as plt
     import numpy as np
     import time

     from sklearn.datasets import fetch_lfw_people
     # Download the data, if not already on disk and load it as numpy arrays
     lfw_people = fetch_lfw_people(min_faces_per_person=40, resize=0.3)
     target = lfw_people.target
     print('target has classes', np.unique(target, return_counts=True))

     # preprocessing -------------------------------------------------------------------------------
     # 121, 144
     cla=[3,17]; test_ratio = 0.25
     
     proc_imag, labels = Wrapper_Dat(cla, lfw_people)
     print("dim of classes", np.unique(labels, return_counts=True))
     dat_train, dat_test, lab_train, lab_test = train_test_split(proc_imag, 
                                                  labels, test_ratio, seed=21203)
     print("dim of train ", dat_train.shape)
     print("train to test ratio ", np.unique(lab_train, return_counts=True), 
                                   np.unique(lab_test, return_counts=True))

     var1_train, var2_train, var1_test, var2_test = Varf(dat_train, dat_test, 
                                        lab_train, lab_test, Wrapper_ker_Gauss, hyper=2)
     print('train set variances', var1_train, var2_train, '\n')
     print('test set variances', var1_test, var2_test, '\n')
     