# Image processing Padded patches --------------------------------------------------------------
def pad_patches(imag, p=3):
     """Extracts patches of any n-dimensional array in place using strides, 
     with stepsize 1
     -----------
     Input
     imag: np array storing images
     p: int; The width (and height) of a patch (odd)
     -----------
     Returns
      patches : strided ndarray
      4n-dimensional array indexing patches with original row and column
     """
     import numpy as np
     from numpy.lib.stride_tricks import as_strided

     # dimension of image and patches
     i_h, i_w = imag.shape
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

     renorm_patches = patches/np.sqrt(np.sum(patches**2, axis=(-1,-2), keepdims=True))

     # pad the patches with p by p arrays of zeors 
     arr_hstack = np.empty((shape_out[0], 1, p, p))
     arr_hstack.fill(np.nan)

     arr_vstack = np.empty((1, shape_out[1]+2, p, p))
     arr_vstack.fill(np.nan)

     pdd_patches = np.hstack((arr_hstack, renorm_patches, arr_hstack))
     #+2 accounting for the extra dim
     pdd_patches = np.vstack((arr_vstack, pdd_patches, arr_vstack)) 

     return pdd_patches

def Wrapper_pad_patches(arr_imag, p=3):
     
     from itertools import starmap, product
          
     return np.array(list(starmap(pad_patches, product(arr_imag, np.array([p])))))


def Wrapper_Dat_5d(cla, dat, p=3):
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
     # cla = [1,2]
     cla_index = np.concatenate([np.where(target == i)[0] for i in cla])
     arr_dat = dat.images[cla_index]
     # scale the original image
     arr_dat = (arr_dat - np.mean(arr_dat, axis=0))/np.std(arr_dat, axis=0)
     proc_imag = Wrapper_pad_patches(arr_dat, p)

     # change the label to +/-1; if input class is 2-dim
     if len(cla) is 2:

          labels = target[cla_index]
          print("class of image", np.unique(labels, return_counts=True))
          labels = [-1 if i==cla[0] else 1 for i in labels]
          # labels[np.where(labels == cla[0])[0]] = 1
          # labels[np.where(labels == cla[1])[0]] = -1

          return proc_imag, np.array(labels)

     elif len(cla) is 1:

          return proc_imag


# Image processing vectorised patches -----------------------------------------------------------
def extra_patches(imag, p=3):
     """Extracts patches of any n-dimensional array in place using strides, 
     with stepsize 1
     -----------
     Input
     imag: np array storing images
     p: int; The width (and height) of a patch (odd)
     -----------
     Returns
      patches : strided ndarray
      2n-dimensional array indexing patches on first n dimensions and
      containing patches on the last n dimensions. 
     """
     import numpy as np
     from numpy.lib.stride_tricks import as_strided

     # dimension of image and patches
     i_h, i_w = imag.shape
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

     # reshape into 
     return patches.reshape(n_patches, p, p)

def dprocess(imag, p=3):
     """Reshape
     ----------
     Input
     arr_dat: 2d array; image to be processed
     p: dimension of square patches
     ----------
     Output
     3d array; each row contains patches of sizes p by p in a vector form
     """
     patches = extra_patches(imag=imag, p=p).reshape(-1, p**2)

     # return normalised vector in each row
     return patches/np.sqrt(np.sum(patches**2, axis=1))[:, np.newaxis]


def Wrapper_dprocess(arr_imag, p=3):
     """Reshape the data into desirable form
     ----------
     Input
     arr_dat: nd array; each row contains one image to be processed
     p: dimension of square patches
     ----------
     Output
     nd array; each row contains patches of sizes p by p in a vector form
     """
     from itertools import starmap, product
     
     patched_dat = list(starmap(dprocess, product(arr_imag, np.array([p]))))
     
     return np.array(patched_dat)


# Train/test split ------------------------------------------------------------------------------
     
def train_test_split(dat, lab, test_ratio=0.25, seed=17671):
     """Split dataset into train and test set
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
     ind_test_neg = rs.rand(neg_dat.shape[0]) < test_ratio
     ind_test_pos = rs.rand(pos_dat.shape[0]) < test_ratio

     # split into test/train classes
     dat_test = np.concatenate((neg_dat[ind_test_neg], pos_dat[ind_test_pos]))
     dat_train = np.concatenate((neg_dat[~ind_test_neg], pos_dat[~ind_test_pos]))

     lab_test = np.concatenate((neg_lab[ind_test_neg], pos_lab[ind_test_pos]))
     lab_train = np.concatenate((neg_lab[~ind_test_neg], pos_lab[~ind_test_pos]))

     return dat_train, dat_test, lab_train, lab_test

# Nearest Mean Clustering ---------------------------------------------------------------
def NearestMean(kernel, hyper, arr_train, arr_test, lab_train, lab_test):
     """Find the kernel population mean for two classes, and make classification
     on the test set with 0-1 loss
     -----------
     Input:
     kernel: function to compute the kernel martix
     hyper: float; e.g. sigma in Gaussian kernel
     -----------
     Output:
     pred_class: predicted label for the test set;
     conf_mat: confusion matrix of the true label and the predicted;
     error: missclassification error
     """

     X_arr_train = arr_train[np.where(lab_train==1)[0]]
     Y_arr_train = arr_train[np.where(lab_train==-1)[0]]

     mu_norm = kernel(Y_arr_train, Y_arr_train, hyper)
     nu_norm = kernel(X_arr_train, X_arr_train, hyper)

     # predict for training data
     pred_class = np.sign(1/2 * (np.mean(mu_norm) - np.mean(nu_norm)) 
                    - np.mean(kernel(arr_test, Y_arr_train, hyper), axis=1) 
                    + np.mean(kernel(arr_test, X_arr_train, hyper), axis=1))

     # confusion matrix
     conf_mat = pd.crosstab(pd.Series(lab_test, name="actual"), 
                              pd.Series(pred_class.astype('int'), name="predict"))
     misclass = (conf_mat.iloc[0,1]+conf_mat.iloc[1,0])/conf_mat.values.sum()
     # print(conf_mat, "\n")

     return {'pred_class': pred_class, 'conf_mat': conf_mat, 'error': misclass}

def Wrapper_Dat(cla, dat, p=3):
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
     # cla = [1,2]
     cla_index = np.concatenate([np.where(target == i)[0] for i in cla])
     arr_dat = dat.images[cla_index]
     # scale the original image
     arr_dat = (arr_dat - np.mean(arr_dat, axis=0))/np.std(arr_dat, axis=0)
     proc_imag = Wrapper_dprocess(arr_dat, p)

     # change the label to +/-1; if input class is 2-dim
     if len(cla) is 2:

          labels = target[cla_index]
          print("class of image", np.unique(labels, return_counts=True))
          labels = [-1 if i==cla[0] else 1 for i in labels]
          # labels[np.where(labels == cla[0])[0]] = 1
          # labels[np.where(labels == cla[1])[0]] = -1

          return proc_imag, np.array(labels)

     elif len(cla) is 1:

          return proc_imag

# Kernel --------------------------------------------------------------------------------
def ker_Gauss2d(X, Y, l):
     """Compute image wise 2d Gaussian kernel, then take mean

     """
     n = X.shape[0]
     m = Y.shape[0]

     # binomial formula ~ Smola's Blog
     tmp = np.tile(np.sum(X**2, axis=1, keepdims=True), m)
     tmp += np.tile(np.sum(Y**2, axis=1), (n,1))
     tmp -= 2 * X @ Y.T

     return np.mean(np.exp(-tmp/(l**2)))

def ker_Imag(X_array, Y_array, l):
     
     from itertools import starmap, product
     n = X_array.shape[0]
     m = Y_array.shape[0]

     ker = np.array(list(starmap(ker_Gauss2d, product(X_array, Y_array, np.array([l])))))
     print("dimension of kernel ", ker.reshape(n,m).shape)

     return ker.reshape(n, m)

def ker_Gauss(u, v, l):
     return np.exp(-np.sum((u - v)**2)/l**2)

def ker_Gauss50(X, Y, l):
     # X and Y are of same shape
     # image processed to be p=3
     from itertools import product

     ker = []

     for i in range(1, X.shape[0]-1):
          for j in range(1, X.shape[1]-1):
               ker += [ker_Gauss(X[i, j, :, :], Y[i, j, :, :], l)]
               for p in [-1, 1]:
                    ker += [ker_Gauss(X[i, j, :, :], Y[i, j+p, :, :], l)]
                    # ker += ker_Gauss(X[i][j], Y[i][j+1], l)
                    ker += [ker_Gauss(X[i, j, :, :], Y[i+p, j, :, :], l)]
                    # ker += ker_Gauss(X[i][j], Y[i+1][j], l)
     return np.nanmean(np.array(ker))
     
     
def Wrapper_ker_Gauss50(X_array, Y_array, l):
     from itertools import starmap, product

     n = X_array.shape[0]; m = Y_array.shape[0]; N = X_array.shape[1] * X_array.shape[2]

     ker_class = list(starmap(ker_Gauss50, product(X_array, Y_array, np.array([l]))))

     return np.array(ker_class).reshape(n, m)/5**2

     
     
# Fitting ---------------------------------------------------------------------------------
def NearestMean(kernel, hyper, arr_train, arr_test, lab_train, lab_test):
     """Find the kernel population mean for two classes, and make classification
     on the test set with 0-1 loss
     -----------
     Input:
     kernel: function to compute the kernel martix
     hyper: float; e.g. sigma in Gaussian kernel
     -----------
     Output:
     pred_class: predicted label for the test set;
     conf_mat: confusion matrix of the true label and the predicted;
     error: missclassification error
     """

     X_arr_train = arr_train[np.where(lab_train==1)[0]]
     Y_arr_train = arr_train[np.where(lab_train==-1)[0]]

     mu_norm = kernel(Y_arr_train, Y_arr_train, hyper)
     nu_norm = kernel(X_arr_train, X_arr_train, hyper)

     # predict for training data
     pred_class = np.sign(1/2 * (np.mean(mu_norm) - np.mean(nu_norm)) 
                    - np.mean(kernel(arr_test, Y_arr_train, hyper), axis=1) 
                    + np.mean(kernel(arr_test, X_arr_train, hyper), axis=1))

     # confusion matrix
     conf_mat = pd.crosstab(pd.Series(lab_test, name="actual"), 
                              pd.Series(pred_class.astype('int'), name="predict"))
     misclass = (conf_mat.iloc[0,1]+conf_mat.iloc[1,0])/conf_mat.values.sum()
     # print(conf_mat, "\n")

     return {'pred_class': pred_class, 'conf_mat': conf_mat, 'error': misclass}


def Wrapper_Plot(kernel_dic, hyper_dic, arr_train, arr_test, lab_train, lab_test, p):
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

     for j in range(len(kernel_dic)):
          ker_name, kernel = list(kernel_dic.items())[j]
          hyper_name, hyper_lis = list(hyper_dic.items())[j]
          error_lis = []

          for i in hyper_lis:
               pred_class = NearestMean(kernel, i, arr_train, arr_test, lab_train, lab_test)
               error_lis += [pred_class['error']]

          min_error = min(error_lis)
          argmin = hyper_lis[error_lis.index(min_error)]

          plt.figure()
          plt.plot(hyper_lis, error_lis, label='Error')
          plt.hlines(min_error, xmin=hyper_lis.min(), xmax=hyper_lis.max(), 
                    label='Min Error {}'.format(round(min_error,3)))
          plt.title('Nearest Mean Clustering with {}'.format(ker_name))
          plt.xlabel('Hyperparameter {a}/ argmin {a}={b}'.format(a=hyper_name, b=round(argmin,3)))
          plt.ylabel('Error of Misclassification/p={}'.format(p))
          plt.legend()
          plt.savefig('{a}_{b}.png'.format(a=ker_name, b=p))
          plt.show()
          plt.close()

# Q1 b) Distance to class population mean
def dist_to_mean(dat, kernel, hyper):
     """Find the distance between an image to empirical mean embedding in the feature space,
     which is then ranked in decreasing order.
     """
     ker_eval = kernel(dat, dat, hyper) 
     ker_dist = (np.diag(ker_eval) - 2 * np.mean(ker_eval, axis=1)
                                   + np.mean(ker_eval))

     return np.sqrt(ker_dist)

def Wrapper_HeatMap(kernel_dic, hyper_dic, dat, p):
     """Wrapper function: visualise the change of order of rank with varying hyper parameters/
     """
     import seaborn as sns

     for j in range(len(kernel_dic)):
          ker_name, kernel = list(kernel_dic.items())[j]
          hyper_name, hyper_lis = list(hyper_dic.items())[j]
          dist_lis = []

          for i in hyper_lis:
               dist_lis += [(-dist_to_mean(dat, kernel, i)).argsort()]
          
          dist_arr = np.array(dist_lis).T

          plt.figure()
          ax = sns.heatmap(dist_arr, cbar_kws={'label': 'Rank of distance (descending)'}
                                   , cmap='YlGnBu')
          ax.set_xticklabels(np.round(hyper_lis,2), rotation=45, fontsize=8)
          ax.set_yticklabels(ax.get_yticklabels(), rotation=45, fontsize=8)
          ax.set_ylabel('Indexed Images/p={}'.format(p))
          ax.set_xlabel('Hyperparameter {}'.format(hyper_name))
          ax.set_title('Change of Distance to Class Mean with {}'.format(ker_name))
          plt.savefig('{a}_{b}_dist.png'.format(a=ker_name, b=p))
          plt.show()
          plt.close()



     # from itertools import product

     # for i, X in enumerate(X_array[:-1]):
     #      # i start from i=1 to 
     #      for j, Y in enumerate(Y_array):

if __name__=='__main__':
     import matplotlib.pyplot as plt
     import numpy as np
     import time

     # can we use this function, but what should be the `resize` argument? By default?
     from sklearn.datasets import fetch_lfw_people
     # Download the data, if not already on disk and load it as numpy arrays
     lfw_people = fetch_lfw_people(min_faces_per_person=40, resize=0.3)
     target = lfw_people.target

     # testing with classes size 77, 42 ------------------------------------------------
     cla=[0,1]; test_ratio = 0.25

     p = 3 # patch size
     # 5d version
     proc_imag_, labels_ = Wrapper_Dat_5d(cla, lfw_people, p)
     print("image label", np.unique(labels_, return_counts=True))
     dat_train_, dat_test_, lab_train_, lab_test_ = train_test_split(proc_imag_, 
                                                  labels_, test_ratio, seed=2823)
     # 3d version
     proc_imag, labels = Wrapper_Dat(cla, lfw_people, p)
     dat_train, dat_test, lab_train, lab_test = train_test_split(proc_imag, 
                                                  labels, test_ratio, seed=2823)
     print("dim of train ", dat_train.shape)


     # print(dat_train_)
     # compare -------------------------------------------------------------------------
     # non-vectorised; for loops over 50% overlapping patches (really slow, how to improve?)
     t0 = time.time()
     ker_ = Wrapper_ker_Gauss50(dat_test_, dat_test_, 2) # 5d array
     t1 = time.time()
     print("time taken to run, for loops", t1-t0)
     print("for loop ker ", ker_)

     t0 = time.time()
     ker = ker_Imag(dat_test, dat_test, 2) # 5d array
     t1 = time.time()
     print("time taken to run, for loops", t1-t0)
     print("for loop ker ", ker)

     # ----------------------------------------------------------------------------------
     cla = [0,1]; test_ratio = 0.25
     kernel_dic = {'Gaussian kernel': ker_Imag
                    # , 'Polynomial kernel': ker_Poly3d
                    }
     hyper_dic = {'$\sigma$': np.linspace(0.1,5,num=20)
                    # , 'c': np.linspace(0,5,num=20)
                    }

     
     p = 3
     proc_imag, labels = Wrapper_Dat(cla, lfw_people, p)
     # print("image label", np.unique(labels, return_counts=True))
     dat_train, dat_test, lab_train, lab_test = train_test_split(proc_imag, 
                                                  labels, test_ratio, seed=2823)
     print("proc_imag size ", proc_imag.shape, '\n')
     print("dim train", dat_train.shape, '\n')
     print("dim test", dat_test.shape, '\n')
     print("train image label", np.unique(lab_train, return_counts=True), '\n')
     print("test image label", np.unique(lab_test, return_counts=True), '\n')

     Wrapper_Plot(kernel_dic, hyper_dic, dat_train, dat_test, lab_train, lab_test, p)