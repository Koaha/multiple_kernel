#!/usr/bin/env python3
"""Implementation of Online Support Vector Regression (OSVR) as library for a class project in 16-831
Statistical Techniques in Robotics.

Requires Python 3.5
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
import numpy as np
import pandas as pd
import pickle

from copy import copy
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
import time
sys.path.append("..")
from kernels.RBF import RBF, Linear
from utilities.gram import gram
from utilities import *

class LaskovOnlineSVR(BaseEstimator,ClassifierMixin):
    """
    :param prune : 0 no prune
                 : 1 pre prune
                 : 2 post prune
    """
    def __init__(self, C=1, eps=1e-3, kernel =None, kernelParam=None, bias=0, prune=False,debug=False,verbose=0):
        # Configurable Parameters
        self.prune = prune
        # self.numFeatures = numFeatures
        self.C = C
        self.eps = eps
        self.kernelParam = kernelParam
        self.bias = bias
        self.debug = debug
        self.verbose = verbose
        print('SELF', self.C, self.eps, self.kernelParam)
        # Algorithm initialization
        self.numSamplesTrained = 0
        self.theta = np.array([])

        # Samples X (features) and Y (truths)
        self.X = np.array([])
        self.Y = np.array([])
        # Working sets, contains indices pertaining to X and Y
        self.supportSetIndices = np.array([])
        self.errorSetIndices = np.array([])
        self.remainderSetIndices = np.array([])
        self.R = np.array([[]])

        if kernel == None:
            self.kernel = RBF()
        else:
            self.kernel = kernel

    def learn(self, newSampleX, newSampleY):
        #Line 1 Add X,Y to the training set
        self.numSamplesTrained +=1 #increase the sample size
        self.X = np.vstack((self.X, newSampleX))
        self.Y = np.vstack((self.Y, newSampleY))
        self.weights = np.vstack((self.weights, 0))
        i = self.numSamplesTrained - 1
        # The new set with additional new Sample input
        h_xc = self.computeMargin(newSampleX,newSampleY)
        #H(xc) is the Margin at index xc = i (latest sample)
        if (abs(h_xc) <= self.eps):
            # Line 3 add new sample to the remaining set and exit
            if self.verbose == 1:
                print('Adding new sample {0} to remainder set, within eps.'.format(i))
            self.remainderSetIndices = np.hstack((self.remainderSetIndices,i))
            return

        addNewSample = False
        KKT_violation = True
        iterations = 0
        # Let q= sign(-(h(xc)) be the sign that ∆θc will take
        q = np.sign(h_xc)

        while KKT_violation:
            beta,gamma = self.computeBetaGamma(i)
            iterations += 1
            # Line6.1 Update beta gamma at the new sample i
            # Using equation 10 & 12
            beta,gamma = self.computeBetaGamma(i)
            #Line6.2 Find least variations

            KKT_violation = self.bookeepingProcedure()
        return

    def bookingProcedure(self):

        # Check the moving case of new sample C
        # Case 1: Moving by the margin change from remaining set to support set

        # Case 2: Moving by theta from error set to support set

        #===========================================
        #== Checking the moving of existing sample==
        #===========================================
        # Case 3:

        return

    def fit(self,X,Y):
        samples = []
        run_time = []
        start = time.time()
        self.numFeatures = X.shape[1]
        X = np.array(X)
        Y = np.array(Y)
        Y[Y==0] = -1
        for i in tqdm(range(X.shape[0])):
            self.learn(X[i, :], [Y[i]])
            if self.verbose == 1:
                print('%%%%%%%%%%%%%%% Data point {0} %%%%%%%%%%%%%%%'.format(i))
            if self.prune == True:
                self.prune_vector()
            stop = time.time()
            samples.append(i)
            run_time.append(stop - start)
        # df = pd.DataFrame({'samples': samples, 'run_time': run_time})
        # df.to_csv(self.__class__.__name__ + "_runtime.csv")

    # Compute  output kernels K(x_j,x) with size (n x m)
    # n: number of support vectors in support set
    # m: number of samples for prediction
    def computeKernelOutput(self,support_set,input_sample):
        if (self.kernel.__class__.__name__ == 'RBF'):
            K = self.kernel.__call__(support_set, input_sample,gamma=self.kernelParam)
        else:
            K = gram()
            K = K.__call__(support_set, input_sample, self.kernel)
        return K

    # Compute the prediction or F(x) of the set
    # of up-to-date samples set ; sampleX size: n x features
    def predict(self, newSampleX):
        set_ES_indices = np.hstack((self.supportSetIndices,self.errorSetIndices))
        if (set_ES_indices.size != 0):
            set_ES = self.X[set_ES_indices]
            K = self.computeKernelOutput(set_ES,newSampleX)
            return K.T.dot(self.theta.reshape(-1, 1)) + self.bias
        return np.zeros_like(newSampleX) + self.bias

    # Line 3 compute the margin H(x) by taking
    # the differences with the prediction F(x)
    def computeMargin(self, newSampleX, newSampleY):
        return self.predict(newSampleX) -newSampleY # input is the set X + new instance X


    def computeQ(self,support_set,sample_set):
        if support_set.size > 0:
            return self.computeKernelOutput(support_set, sample_set)
        return np.array([[]])

    def computeBetaGamma(self,i):
        support_set = self.X[self.supportSetIndices]
        c = self.X[i]
        Qsi = self.computeQ(support_set,c)
        if self.supportSetIndices.size == 0:
            beta =  np.array([])
        else:
            beta = -self.R @ np.vstack((1,Qsi))

        non_support_set_indices = np.hstack((self.remainderSetIndices,self.errorSetIndices))
        non_support_set = self.X[non_support_set_indices]

        Qnc = self.computeQ(non_support_set, c)  # compute kernel of all samples vs new samples
        Qns = self.computeQ(non_support_set, support_set)  # compute kernel of all samples vs support set

        if non_support_set.size == 0:
            gamma = np.array(np.ones_like(Qnc))
        else:
            gamma = Qnc + np.hstack((np.ones(Qnc.shape),Qns)) @ beta

        # Correct for NaN
        beta[np.isnan(beta)] = 0
        gamma[np.isnan(gamma)] = 0

        return beta, gamma

    def computeLc1(selfs):
        return

    def computeLc2(self):
        return

def main(argv):
    # Test of Online SVR algorithm
    debug = True if len(argv) > 1 and argv[1] == 'debug' else False

    readX = pd.read_csv('toyX.csv')
    testSetX = np.array(readX)
    readY = pd.read_csv('toyY.csv')
    testSetY = np.array(readY)

    OSVR = LaskovOnlineSVR(numFeatures=testSetX.shape[1], C=10,
                     eps=0.1, kernelParam=None, bias=0, debug=debug,prune=False)

    for i in range(testSetX.shape[0]):
        print('%%%%%%%%%%%%%%% Data point {0} %%%%%%%%%%%%%%%'.format(i))
        OSVR.learn(testSetX[i, :], testSetY[i])

    save_model('online-svr.pkl', OSVR)
    # Predict stuff as quick test
    testX = np.array([[0.15, 0.1], [0.25, 0.2]])
    testY = np.sin(2 * np.pi * testX)
    print('testX:', testX)
    print('testY:', testY)

    OSVR = load_model('online-svr.pkl')
    PredictedY = OSVR.predict_(np.array([testX[0, :]]))
    Error = OSVR.computeMargin(testX[0], testY[0])
    print('PredictedY:', PredictedY)
    print('Error:', Error)
    return OSVR

def save_model(fname, model):
    with open(fname, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
    return model

def load_model(fname):
    with open(fname, 'rb') as input:
        model = pickle.load(input)
    return model

if __name__ == '__main__':
    # main(sys.argv)
    print('run main plz')
