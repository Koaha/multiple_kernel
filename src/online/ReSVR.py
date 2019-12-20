#!/usr/bin/env python3
"""Implementation of Online Support Vector Regression (OSVR) as library for a class project in 16-831
Statistical Techniques in Robotics.

Requires Python 3.5
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys,os
import numpy as np
import pandas as pd
import pickle

from copy import copy
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
import time
sys.path.append("..")
from src.kernels.kernels import RBF,Linear
from src.utilities.gram import gram
from src.utilities import *

class NySVM(BaseEstimator,ClassifierMixin):
    """
    :param prune : 0 no prune
                 : 1 pre prune
                 : 2 post prune
    """
    def __init__(self,numFeatures, C=1, eps=1e-3, kernel =None, kernelParam=None, bias=0, prune=False,debug=False,verbose=0):
        # Configurable Parameters
        self.prune = prune
        self.numFeatures = numFeatures
        self.C = C
        self.eps = eps
        self.kernelParam = kernelParam
        self.bias = bias
        self.debug = debug
        self.verbose = verbose
        # Algorithm initialization
        self.numSamplesTrained = 0
        self.omega = np.array([])

        # Samples X (features) and Y (truths)
        self.X = np.array([])
        self.Y = np.array([])
        # Working sets, contains indices pertaining to X and Y
        self.supportSetIndices = np.array([])
        self.errorSetIndices = np.array([])
        self.remainderSetIndices = np.array([])
        self.R = np.array([])

        if kernel == None:
            self.kernel = RBF()
        else:
            self.kernel = kernel

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
    def predict_(self, newSampleX,margin_flag = False):
        if (self.numSamplesTrained>0):
            support_vector = self.X[self.supportSetIndices]
            if margin_flag:
                support_vector = self.X
            K = self.computeKernelOutput(support_vector,newSampleX)
            return K.T@self.omega.reshape(-1,1) + self.bias
        else:
            return np.zeros_like(newSampleX) + self.bias

    """
    Case of margin:
    1) h(x_i) >= eps                -> omega_i= -C
    2) h(x_i) == eps                -> -C < omega_i < 0
    3) -eps <= h(x_i) <= eps        -> omega_i= 0
    4) h(x_i) == -eps               ->  0 < omega_i < C
    5) h(x_i) <= -eps               -> omega_i=0
    """
    # Line 3 compute the margin H(x) by taking
    # the differences with the prediction F(x)
    def computeMargin(self, newSampleX, newSampleY):
        fx = self.predict_(newSampleX,margin_flag=True)  # input is the set X + new instance X
        newSampleY = np.array(newSampleY)
        return fx - newSampleY

    def computeQ_append(self):
        if self.supportSetIndices.size == 0:
            return np.array([[0]])
        support_vector = self.X[self.supportSetIndices]
        Q = self.computeKernelOutput(support_vector, support_vector)
        row, col = Q.shape
        Q_append_row = np.vstack((np.ones(row), Q))
        Q_append_row_col = np.hstack((np.ones(col + 1), Q_append_row))
        Q_append_row_col[0, 0] = 0
        return Q_append_row_col

    def computeR(self):
        if self.supportSetIndices.size == 0:
            return np.array([[0]])
        Q_append_row_col = self.computeQ_append()
        self.R = np.linalg.inv(Q_append_row_col)
        return self.R

    def computeBeta(self,new_sample):
        support_vector = self.X[self.supportSetIndices]
        Qsc = self.computeKernelOutput(support_vector,new_sample)
        Qsc_append = np.vstack((1,Qsc.reshape(-1,1)))
        beta = -self.R@Qsc_append
        return beta

    def initialization(self,newSampleX, newSampleY):
        omega_c = 0
        hxc = self.computeMargin(newSampleX, newSampleY)


DATA_PATH = "../../data"
TOY_DATA_PATH = os.path.join(DATA_PATH,"ToyDS")
PIPE_DATA_PATH = os.path.join(DATA_PATH,"PipeDS")
DC_DATA_PATH = os.path.join(DATA_PATH,"DC")
BINARY_DATA_PATH = os.path.join(DATA_PATH,"binary")
PEGASOS_DATA_PATH = os.path.join(DATA_PATH,"pegasosDS")


def main(argv):
    # Test of Online SVR algorithm
    debug = True if len(argv) > 1 and argv[1] == 'debug' else False

    print(os.getcwd())
    readX = pd.read_csv(os.path.join(DATA_PATH,'toyX.csv'))
    testSetX = np.array(readX)
    readY = pd.read_csv(os.path.join(DATA_PATH,'toyY.csv'))
    testSetY = np.array(readY)

    OSVR = NySVM(numFeatures=testSetX.shape[1], C=10,
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
    main(sys.argv)
    # print('run main plz')
