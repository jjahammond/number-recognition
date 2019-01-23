"""M3C 2018 Homework 2"""

import numpy as np
import matplotlib.pyplot as plt
from m1 import nmodel as nm #assumes that hw2_dev.f90 has been compiled with: f2py -c hw2_dev.f90 -m m1
# May also use scipy, scikit-learn, and time modules as needed

def read_data(tsize=60000):
    """Read in image and label data from data.csv.
    The full image data is stored in a 784 x 70000 matrix, X
    and the corresponding labels are stored in a 70000 element array, y.
    The final 70000-tsize images and labels are stored in X_test and y_test, respectively.
    X,y,X_test, and y_test are all returned by the function.
    You are not required to use this function.
    """
    print("Reading data...") #may take 1-2 minutes
    Data=np.loadtxt('data.csv',delimiter=',')
    Data =Data.T
    X,y = Data[:-1,:]/255.,Data[-1,:].astype(int)%2 #rescale the image, convert the labels to 0s and 1s (For even and odd integers)
    Data = None

    # Extract testing data
    X_test = X[:,tsize:]
    y_test = y[tsize:]

    print("processed dataset")
    return X,y,X_test,y_test
#----------------------------

def snm_test(X,y,X_test,y_test,tsize=60000):
    """Train single neuron model with input images and labels (i.e. use data in X and y), then compute and return testing error in test_error
    using X_test, y_test. The fitting parameters obtained via training should be returned in the 1-d array, fvec_f
    X: training image data, should be 784 x d with 1<=d<=60000
    y: training image labels, should contain d elements
    X_test,y_test: should be set as in read_data above
    omethod=1: use l-bfgs-b optimizer
    omethod=2: use stochastic gradient descent
    input: tuple, set if and as needed
    """

    X_train = X[:,:tsize]
    y_train = y[:tsize]

    n = X_train.shape[0]
    d = X_train.shape[1]
    alpha = 0.1
    T = y_test.size
    fvec_guess = np.random.randn(n+1) #initial fitting parameters

    a = np.zeros(T)

    nm.data_init(n,d)
    nm.nm_x = X_train
    nm.nm_y = y_train


    print("Training SNM")
    fvec_f = nm.sgd(fvec_guess,n,0,d,alpha)

    print("Testing result")
    for k in range(T):
        z = sum(np.multiply(fvec_f[:-1], X_test[:,k])) + fvec_f[-1]
        a[k] = round(1 / (1 + np.exp(-z)))

    e = a - y_test
    Ncorrect = np.count_nonzero(e==0)

    test_error = 1 - Ncorrect/T #Modify to store testing error; see neural network notes for further details on definition of testing error
    print(test_error)

    return fvec_f,test_error,a
#--------------------------------------------

def nnm_test(X,y,X_test,y_test,m=4,tsize=60000):
    """Train neural network model with input images and labels (i.e. use data in X and y), then compute and return testing error (in test_error)
    using X_test, y_test. The fitting parameters obtained via training should be returned in the 1-d array, fvec_f
    X: training image data, should be 784 x d with 1<=d<=60000
    y: training image labels, should contain d elements
    X_test,y_test: should be set as in read_data above
    m: number of neurons in inner layer
    omethod=1: use l-bfgs-b optimizer
    omethod=2: use stochastic gradient descent
    input: tuple, set if and as needed
    """
    X_train = X[:,:tsize]
    y_train = y[:tsize]

    n = X_train.shape[0]
    d = X_train.shape[1]

    alpha = 0.1
    T = y_test.size
    fvec_guess = np.random.randn(m*(n+2)+1) #initial fitting parameters

    a_outer = np.zeros(T)

    #Add code to train NNM and evaluate testing error, test_error
    nm.data_init(n,d)
    nm.nm_x = X_train
    nm.nm_y = y_train

    #Add code to train SNM and evaluate testing test_error
    print("Training NNM")
    fvec_f = nm.sgd(fvec_guess,n,m,d,alpha)

    w_inner = fvec_f[:n*m].reshape(n,m).T
    b_inner = fvec_f[n*m:n*m+m]
    w_outer = fvec_f[n*m+m:n*m+2*m]
    b_outer = fvec_f[n*m+2*m]

    print("Testing NNM")
    for k in range(T):
        z_inner = b_inner + sum(np.multiply(w_inner, X_test[:,k]).T)
        a_inner = 1 / (1 + np.exp(-z_inner))

        z_outer = b_outer + sum(np.multiply(w_outer,a_inner))
        a_outer[k] = round(1 / (1 + np.exp(-z_outer)))

    e = a_outer - y_test
    Ncorrect = np.count_nonzero(e==0)

    test_error = 1 - Ncorrect/T #Modify to store testing error; see neural network notes for further details on definition of testing error
    #output = (None) #output tuple, modify as needed
    print(test_error)

    return fvec_f,test_error
#--------------------------------------------

def nm_analyze():
    """ Analyze performance of single neuron and neural network models
    on even/odd image classification problem
    Add input variables and modify return statement as needed.
    Should be called from
    name==main section below
    """

    return None
#--------------------------------------------

def display_image(X):
    """Displays image corresponding to input array of image data"""
    n2 = X.size
    n = np.sqrt(n2).astype(int) #Input array X is assumed to correspond to an n x n image matrix, M
    M = X.reshape(n,n)
    plt.figure()
    plt.imshow(M)
    return None
#--------------------------------------------
#--------------------------------------------


if __name__ == '__main__':
    #The code here should call analyze and generate the
    #figures that you are submitting with your code
    output = nm_analyze()
