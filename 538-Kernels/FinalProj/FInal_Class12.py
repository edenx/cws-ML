import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# Import MNIST data all classes --------------------------------------------------------------------
n_epochs = 1
batch_size_train = 64 + 200
batch_size_test = 1000 # fixed for 300 test case

# Importing MNIST data: first 2 classes -----------------------------------------------
from torch.utils.data._utils.collate import default_collate
def my_collate(batch):
    modified_batch = []
    for item in batch:
        image, label = item
        if label==1 or label==2:
            modified_batch.append(item)
    return default_collate(modified_batch)

train_loader12 = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data12_train.pt', train=True, download=True, 
    transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ])),
    batch_size=batch_size_train, shuffle=True, collate_fn=my_collate)

test_loader12 = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data12_test.pt', train=False, download=True, 
    transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ])),
    batch_size=batch_size_train, shuffle=True, collate_fn=my_collate)

# Building the network ----------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28, 1)
        self.fc2 = nn.Linear(28, 300)

    def forward(self, x):
        # flatten from 28x28x1 to 784
        x = self.fc1(x)
        x = x.view(-1, 28)
        x = F.relu(self.fc2(x))
        
        return x

# intialise weight
def init_weights(m):
     classname = m.__class__.__name__
     # for every Linear layer in a model..
     if classname.find('Linear') != -1:
          # get the number of the inputs
          n = m.in_features
          print("number of feature is ", n)
          if(n==28):
               m.weight.data.normal_(0, 1)
          # get the numnber of channels
          else:
               m.weight.data.normal_(0, 1/(n*2))
          m.bias.data.fill_(0)
     

# get NN representation layer -----------------------------------------------------
def getFeature(data_loader, nbatch=1):
     network.train()

     random_seed = 13223
     torch.backends.cudnn.enabled = False
     torch.manual_seed(random_seed)

     n = 0
     imag_vec = torch.Tensor()
     feature = torch.Tensor()
     label = torch.LongTensor()

     for batch_idx, (data, target) in enumerate(data_loader):
          n += 1
          # normalise the data
          norm_dat = data/torch.norm(data, p=2, dim=(1,2), keepdim=True)
          norm_dat[torch.isnan(norm_dat)] = 0

          imag_vec = torch.cat((imag_vec, norm_dat.view(-1, 784)))
          
          feature = torch.cat((feature, network(norm_dat)))
          label = torch.cat((label, target))

          if n == nbatch:
               # change the labels of second class to 0
               label[label==2] = 0
               # check the proportion of classes
               check_labs = label.data.numpy()
               print(np.unique(check_labs, return_counts=True))

               return imag_vec, feature, label


# ReLU kernel arccos0 -----------------------------------------------------------------
def Kernel_relu(p): return (torch.sqrt(1-p**2) + (np.pi - torch.acos(p)) * p)/np.pi

def centering(ker):
     n, m = ker.shape
     return ((torch.eye(n) - torch.ones((n,n))/n) @ ker @ (torch.eye(m) - torch.ones((m,m))/m))

# dat_train, dat_test, lab_train, lab_test are tensors
# l the regularisation parameter, corresonding to ||f||<R
def kernelRidge_12(dat_train, dat_test, lab_train, lab_test, if_NN=False):
     n = dat_train.shape[0]
     vlen = dat_train.shape[1]
     print("length of representation is", vlen)

     if if_NN:
          
          # normalise
          norm_dat_train = dat_train/(np.sqrt(vlen*2))
          norm_dat_test = dat_test/(np.sqrt(vlen*2))

          # find gram matrix
          Knn = torch.mm(norm_dat_train, norm_dat_train.t())
          Knm = torch.mm(norm_dat_train, norm_dat_test.t())
          print("NN gram", Knn[1:10, 1:10])
     else:
          # find compositional kernel for the skeleton, 3 hidden layers
          Knn = Kernel_relu(torch.matmul(dat_train, dat_train.t())/28)
          Knm = Kernel_relu(torch.matmul(dat_train, dat_test.t())/28)
          # print(torch.norm(dat_train, p=2, dim=1))
          print("Skeleton kernel", Knn[0:10, 0:10])

     return Knn, Knm

def pred_KRR(l, lab_train, lab_test, Knn, Knm):
     n = lab_train.shape[0]
     # prediction
     pred = lab_train.float() @ torch.inverse(Knn/n + l*torch.eye(n)) @ Knm

     pred[pred<0] = 0
     pred[pred>1] = 1

     # get MSE for test set
     error = torch.mean((lab_test - pred)**2)

     return error.data.numpy()



# plot the test set MSR
def plot_MSR(l_lis, nbatch=1, plot=True):
     imagvec_train, repvec_train, lab_train = getFeature(train_loader12, nbatch=nbatch)
 
     imagvec_test, repvec_test, lab_test = getFeature(test_loader12, nbatch=3)

     Knn_s, Knm_s = kernelRidge_12(imagvec_train, imagvec_test, 
                                             lab_train, lab_test, if_NN=False)
     Knn_w, Knm_w = kernelRidge_12(repvec_train, repvec_test, 
                                             lab_train, lab_test, if_NN=True) 
     cKnn_s = centering(Knn_s); cKnm_s = centering(Knm_s)
     cKnn_w = centering(Knn_w); cKnm_w = centering(Knm_w)


     error_lis_s = []
     error_lis_w = []
     for l in l_lis:
          error_lis_s += [pred_KRR(l, lab_train, lab_test, cKnn_s, cKnm_s)]
          error_lis_w += [pred_KRR(l, lab_train, lab_test, cKnn_w, cKnm_w)]

     temp_s = np.min(np.array(error_lis_s))
     min_error_s = np.round(temp_s.astype(float), decimals=3)
     argmin_error_s = np.round(l_lis[np.argmin(np.array(error_lis_s))], decimals=3)

     temp_w = np.min(np.array(error_lis_w))
     min_error_w = np.round(temp_w.astype(float), decimals=3)
     argmin_error_w = np.round(l_lis[np.argmin(np.array(error_lis_w))], decimals=3)

     mins_w = np.array(error_lis_s)[np.argmin(np.array(error_lis_w))]
     diff_argmins_w = np.round(np.abs(mins_w.astype(float) - min_error_w), decimals=3)
 
     if plot:
          plt.figure()
          plt.plot(l_lis, error_lis_s, label='Compositional Kernel')
          plt.plot(l_lis, error_lis_w, label='NN Representation')
          plt.hlines(min_error_s, xmin=l_lis.min(), xmax=l_lis.max(), 
                         label='Comp:Min Error {a}; Argmin {b}'.format(a=min_error_s, b=argmin_error_s), 
                         linestyle=':')
          plt.hlines(min_error_w, xmin=l_lis.min(), xmax=l_lis.max(), 
                         label='NN:Min Error {a}; Argmin {b}'.format(a=min_error_w, b=argmin_error_w), 
                         linestyle='-.')
          plt.title('Kernel Ridge Regression with Square Loss')
          plt.xlabel('Regularisation parameter $\lambda$/Difference at NN argmin is {}'.format(diff_argmins_w))
          plt.ylabel('test set MSE')
          plt.legend()
          plt.savefig('KRR_test{}.png'.format(nbatch))
          plt.show()
          plt.close() 
     
     return min_error_s, min_error_w, diff_argmins_w

if __name__=='__main__':
     import os
     os.environ['KMP_DUPLICATE_LIB_OK']='True'

     network = Net().apply(init_weights)

     l_lis = np.linspace(-0.01, 10, 300)

     plot_MSR(l_lis, nbatch=12)

     # min_lis_s = []; min_lis_w = []; diff_s_w = []
     # for nbatch in np.arange(1, 25):
     #      s, w, s_w = plot_MSR(l_lis, nbatch=nbatch)
     #      min_lis_s += [s]; min_lis_w += [w]; diff_s_w += [s_w]
     
     # # min_min = np.min(np.array(min_lis))
     # plt.figure()
     # plt.plot(np.arange(1, 25), min_lis_s, label='Compositional Kernel')
     # plt.plot(np.arange(1, 25), min_lis_w, label='NN Representation')
     # plt.title('KRR MSE with increasing train set size')
     # plt.xlabel('Train set size/x60')
     # plt.ylabel('test set MSE')
     # plt.legend()
     # plt.savefig('KRR_nbatch.png')
     # plt.show()
     # plt.close() 

     # plt.figure()
     # plt.plot(np.arange(1, 25), diff_s_w)
     # plt.title('Difference of MSE at argmin Empirical Kernel KRR')
     # plt.xlabel('Train set size/x60')
     # plt.ylabel('test set MSE')
     # plt.legend()
     # plt.savefig('KRR_diff.png')
     # plt.show()
     # plt.close() 


