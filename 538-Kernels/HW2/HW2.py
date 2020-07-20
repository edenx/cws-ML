# Pairwise distances between obervation
# A, B two data frame objects
def pdist(A, B):

    n = A.shape[0]
    m = B.shape[0]

     # binomial formula ~ Smola's Blog
    tmp = (A**2).sum(axis=1).values.reshape(n,1) @ np.ones(m).reshape(1,m)
    tmp += np.ones(n).reshape(n,1) @ (B**2).sum(axis=1).values.reshape(1,m)
    tmp -= 2 * A.values @ B.transpose().values

    return tmp
     
# Gaussian kernel
# A, B: data frame objects storing objects with numeric features
# l: positive real, spatial range
def ker_Gauss(A, B, l):

    dist = pdist(A, B)

    return np.exp(-dist/(l**2))

# Polynomial Kernel
# A, B: data frame objects storing objects with numeric features
# c: positive real, shift
def ker_Poly(A, B, c):

    return (A.values @ B.transpose().values + c)**2

# Nearest Mean Implementation
# hyper: sigma in Gaussian kernel; c in Polynomial kernel.
def NearestMean(kernel, hyper, df_train, df_test, lab_train, lab_test):

     mu_norm = kernel(df_train.loc[lab_train==-1,:], df_train.loc[lab_train==-1,:], hyper)
     nu_norm = kernel(df_train.loc[lab_train==1,:], df_train.loc[lab_train==1,:], hyper)

     # predict for training data
     pred_class = np.sign(1/2 * (np.mean(mu_norm) - np.mean(nu_norm)) 
                    - np.mean(kernel(df_test, df_train.loc[lab_train==-1,:], hyper), axis=1) 
                    + np.mean(kernel(df_test, df_train.loc[lab_train==1,:], hyper), axis=1))

     # confusion matrix
     conf_mat = pd.crosstab(pd.Series(lab_test, name="actual"), 
                              pd.Series(pred_class.astype('int'), name="predict"))
     misclass = (conf_mat.iloc[0,1]+conf_mat.iloc[1,0])/conf_mat.values.sum()

     return {'pred_class': pred_class, 'conf_mat': conf_mat, 'error': misclass}

# Wrapper function to produce the plot
# kernel_dic: dictionary storing kernel name and kernel function pairs
# hyper_dic: dictionary storing hyperparameter name and hyperparameter value pairs
def Wrapper_Plot(kernel_dic, hyper_dic, df_train, df_test, lab_train, lab_test):

     for j in range(len(kernel_dic)):
          ker_name, kernel = list(kernel_dic.items())[j]
          hyper_name, hyper_lis = list(hyper_dic.items())[j]
          error_lis = []

          for i in hyper_lis:
               pred_class = NearestMean(kernel, i, df_train, df_test, lab_train, lab_test)
               error_lis += [pred_class['error']]

          min_error = min(error_lis)
          argmin = hyper_lis[error_lis.index(min_error)]

          plt.figure()
          plt.plot(hyper_lis, error_lis, label='Error')
          plt.hlines(min_error, xmin=hyper_lis.min(), xmax=hyper_lis.max(), 
                    label='Min Error {}'.format(round(min_error,3)))
          plt.title('Nearest Mean Clustering with {}'.format(ker_name))
          plt.xlabel('Hyperparameter {a}/ argmin {a}={b}'.format(a=hyper_name, b=round(argmin,3)))
          plt.ylabel('Error of Misclassification')
          plt.legend()
          plt.savefig('{}.png'.format(ker_name))
          plt.show()
          plt.close()

def Wrapper_Fit(kernel, hyper_lis, df_train, df_test, lab_train, lab_test):
     pred_lis = []
     conf_mat = []
     for i in hyper_lis:
          pred_class = NearestMean(kernel, i, df_train, df_test, lab_train, lab_test)
          conf_mat += [pred_class['conf_mat']]
          pred_lis += [pred_class['pred_class']]
     
     return {'conf_mat': conf_mat, 'pred_lis': pred_lis}


     

if __name__=='__main__':
     # only using Numpy and Panda for data manipulation
     # plt for plotting
     import pandas as pd
     import numpy as np
     import numpy.random as rm
     from matplotlib import pyplot as plt

     data = pd.read_csv('https://web.stanford.edu/~hastie/ElemStatLearn//datasets/spam.data', sep='\n', header=None)

     # Data Formatting --------------------------------------------------------------------------------------------
     df = data[0].str.split(' ', expand=True)

     # change the feature name to di; class name as 'spam' 
     df.columns = np.array(['d{}'.format(i) for i in range(1,58)]+['spam'])

     # change the object type to float; integer for class 'spam'
     df[df.columns[:57]] = df[df.columns[:57]].astype('float')
     df['spam'] = df['spam'].astype('int')

     # change the nominal value in 'spam' to +/-1
     df.spam = np.array(list(map(lambda x: 1 if x==1 else -1, df.spam)))

     # standardise the features to have standard deviation 1 and mean 0
     df_features = df.iloc[:,:-1]
     df_features = (df_features-df_features.mean())/df_features.std()

     labels = df.iloc[:,-1]

     # Split into Test and Train set with random function and seed=176007 --------------------------------------------
     rs = rm.RandomState(seed=176007)
     index = rs.rand(df.shape[0])<0.8
     # the proportion of spam are roughly 0.6 for train and test

     df_train = df_features.loc[index,] 
     df_test = df_features.loc[~index,] 

     lab_train = labels[index].values
     lab_test = labels[~index].values

     # Tuneing Hyperparameter ----------------------------------------------------------------------------------------
     # kernel_dic = {'Gaussian kernel': ker_Gauss, 'Polynomial kernel': ker_Poly}
     # hyper_dic = {'$\sigma$': np.linspace(0.1,20,num=40)
     #           , 'c': np.linspace(0,20,num=40)}

     # Wrapper_Plot(kernel_dic, hyper_dic, df_train, df_test, lab_train, lab_test)

     temp = Wrapper_Fit(ker_Gauss, np.linspace(0.1,20,num=40), df_train, df_test, lab_train, lab_test)
     # print(temp['pred_lis'])

     print(temp['pred_lis'][-1] - temp['pred_lis'][-2])
     print(temp['pred_lis'][-2] - temp['pred_lis'][-3])


