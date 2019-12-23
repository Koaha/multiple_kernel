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
    SUPPORT_SET_FLAG = 0
    ERROR_SET_FLAG = 1
    REMAINDER_SET_FLAG = 2

    def sign_numpy(self,a):
        res = np.sign(a)
        res[res==0] = 1
        return res

    def sign_number(self,a):
        if a >= 0:
            return 1
        return -1

    def save_model(self,fname):
        with open(fname, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        return self

    def load_model(self,fname):
        with open(fname, 'rb') as input:
            self = pickle.load(input)
        return self

    def __init__(self,numFeatures, C=1, eps=1e-3, kernel =None, kernelParam=None, bias=0, prune=False,debug=False,verbose=0):
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
        # Line 1 Add X,Y to the training set
        self.numSamplesTrained += 1  # increase the sample size
        self.X = np.vstack((self.X, newSampleX))
        self.Y = np.vstack((self.Y, newSampleY))
        self.weights = np.vstack((self.weights, 0))
        i = self.numSamplesTrained - 1
        # The new set with additional new Sample input
        H = self.computeMargin(self.X,self.Y)
        h_xc = H[-1]
        # H(xc) is the Margin at index xc = i (latest sample)
        if (abs(h_xc) <= self.eps):
            # Line 3 add new sample to the remaining set and exit
            if self.verbose == 1:
                print('Adding new sample {0} to remainder set, within eps.'.format(i))
            self.remainderSetIndices = np.hstack((self.remainderSetIndices, i))
            return

        addNewSample = False
        KKT_violation = True
        iterations = 0
        # Let q= sign(-(h(xc)) be the sign that ∆θc will take
        q = self.sign_number(-h_xc)

        while KKT_violation:
            beta, gamma = self.computeBetaGamma(i)
            iterations += 1
            # Line6.1 Update beta gamma at the new sample i
            # Using equation 10 & 12
            beta, gamma = self.computeBetaGamma(i)
            # Line6.2 Find least variations
            flag, delta_theta_c, moving_vector_index = self.bookKeepingProcedure(h_xc,gamma,q,beta)
            self.bias, self.theta = self.update_theta_bias(beta,delta_theta_c)
            H = self. update_margin_h(H,gamma,delta_theta_c)

            H, KKT_violation = self.adjustSetByFlag(flag,moving_vector_index,H,beta,gamma)
        return

    def adjustSetByFlag(self,flag,i,H,beta,gamma):
        # add new sample to Support set
        if flag == 0:
            if self.verbose == 1:
                print('Adding new sample {0} to support set.'.format(i))
            H[i] = self.sign_number(H[i]) * self.eps
            self.supportSetIndices = np.hstack((self.supportSetIndices,i))
            self.R = self.addSampleToR(i, self.SUPPORT_SET_FLAG, beta, gamma)
            return H, True
        # add new sample to Error set
        elif flag == 1:
            if self.verbose == 1:
                print('Adding new sample {0} to error set.'.format(i))
            self.theta[i] = self.sign_number(self.weights[i]) * self.C
            self.errorSetIndices = np.hstack((self.errorSetIndices, i))
            return H, True
        #remove sample from support set
        elif flag == 2:
            self.theta[i] = self.moveToCloserBound(self.theta[i])
            self.R = self.removeSampleFromR(i)
            if self.theta[i] == 0: # move from support to remainder set
                self.remainderSetIndices = np.hstack((self.remainderSetIndices,i))
            else: # move from support to error set
                self.errorSetIndices = np.hstack((self.errorSetIndices, i))
            remove_index = np.argwhere(self.supportSetIndices == i)
            self.supportSetIndices = np.delete(self.supportSetIndices, remove_index)
        # move sample from Error set to Support set
        elif flag == 3:

    def removeSampleFromR(self,i):

    def moveToCloserBound(self,value):
        res = 0
        if np.abs(value) > self.C/2:
            return self.sign_number(value) * self.C
        return res

    def addSampleToR(self,sample_index, set_flag, beta, gamma):
        """

        :param sample_index:
        :param set_flag: 0 = support; 1= error; 2= remainder
        :param beta:
        :param gamma:
        :return:
        """
        X = np.array(self.X)
        sampleX = X[sample_index, :]
        sampleX.shape = ((int)(sampleX.size / self.numFeatures), self.numFeatures)
        # Add first element
        if self.R.shape[0] <= 1:
            Rnew = np.ones([2, 2])
            Rnew[0, 0] = -self.computeKernelOutput(sampleX, sampleX)
            Rnew[1, 1] = 0
        if set_flag == self.SUPPORT_SET_FLAG:
            # add a column and row of zeros onto right/bottom of R
            r, c = self.R.shape
            R_extend_column = np.append(self.R, np.zeros([r, 1]), axis=1)
            R_extend_column_row = np.append(R_extend_column, np.zeros([1, c + 1]), axis=0)
            beta_extend = np.hstack((beta,1))
            Rnew = R_extend_column_row + 1/gamma[sample_index]* beta_extend @ beta_extend
            return Rnew

    def update_theta_bias(self, beta, delta_theta_c):
        delta_beta_theta_vector = beta*delta_theta_c
        updated_bias = self.bias + delta_beta_theta_vector[0]
        updated_theta = self.theta + delta_beta_theta_vector[1:]
        return updated_bias, updated_theta

    def update_margin_h(self, H,gamma, delta_theta_c):
        set_N_indices = np.concatenate((self. errorSetIndices,self.remainderSetIndices))
        H[set_N_indices] = H[set_N_indices] + gamma*delta_theta_c
        return H

    def bookKeepingProcedure(self,h_xc,gamma,q,beta):
        # Check the moving case of new sample C
        # Case 1: Moving by the margin change from remaining set to support set
        Lc1 = self.computeLc1(h_xc,gamma,q)
        # Case 2: Moving by theta from error set to support set
        Lc2 = self.computeLc2(q)
        # ===========================================
        # == Checking the moving of existing sample==
        # ===========================================
        # Case 3 - Support Set movement
        Ls, Ls_min, Ls_index = self.computeLs(q,beta)
        # Case 4 - Error Set movement
        Le, Le_min, Le_index = self.computeLs(q, beta)
        # Case 5 - Remainder Set movement
        Lr, Lr_min, Lr_index = self.computeLs(q, beta)

        c = self.numSamplesTrained-1
        moving_vector_index_list = np.array([c,c, Ls_index, Le_index, Lr_min])
        moving_cases = np.array([Lc1, Lc2, Ls_min, Le_min, Lr_min])
        flag = np.argmin(moving_cases)
        delta_theta_c = q*moving_cases[flag]
        moving_vector_index = moving_vector_index_list[flag]
        return flag, delta_theta_c, moving_vector_index

    def fit(self, X, Y):
        samples = []
        run_time = []
        start = time.time()
        self.numFeatures = X.shape[1]
        X = np.array(X)
        Y = np.array(Y)
        Y[Y == 0] = -1
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
    def computeKernelOutput(self, support_set, input_sample):
        if (self.kernel.__class__.__name__ == 'RBF'):
            K = self.kernel.__call__(support_set, input_sample, gamma=self.kernelParam)
        else:
            K = gram()
            K = K.__call__(support_set, input_sample, self.kernel)
        return K

    # Compute the prediction or F(x) of the set
    # of up-to-date samples set ; sampleX size: n x features
    def predict(self, newSampleX):
        set_ES_indices = np.hstack((self.supportSetIndices, self.errorSetIndices))
        if (set_ES_indices.size != 0):
            set_ES = self.X[set_ES_indices]
            K = self.computeKernelOutput(set_ES, newSampleX)
            return K.T.dot(self.theta.reshape(-1, 1)) + self.bias
        return np.zeros_like(newSampleX) + self.bias

    # Line 3 compute the margin H(x) by taking
    # the differences with the prediction F(x)
    def computeMargin(self, newSampleX, newSampleY):
        return self.predict(newSampleX) - newSampleY  # input is the set X + new instance X

    def computeQ(self, support_set, sample_set):
        if support_set.size > 0:
            return self.computeKernelOutput(support_set, sample_set)
        return np.array([[]])

    def computeBetaGamma(self, i):
        support_set = self.X[self.supportSetIndices]
        c = self.X[i]
        Qsi = self.computeQ(support_set, c)
        if self.supportSetIndices.size == 0:
            beta = np.array([])
        else:
            beta = -self.R @ np.vstack((1, Qsi))

        non_support_set_indices = np.hstack((self.remainderSetIndices, self.errorSetIndices))
        non_support_set = self.X[non_support_set_indices]

        Qnc = self.computeQ(non_support_set, c)  # compute kernel of all samples vs new samples
        Qns = self.computeQ(non_support_set, support_set)  # compute kernel of all samples vs support set

        if non_support_set.size == 0:
            gamma = np.array(np.ones_like(Qnc))
        else:
            gamma = Qnc + np.hstack((np.ones(Qnc.shape), Qns)) @ beta

        # Correct for NaN
        beta[np.isnan(beta)] = 0
        gamma[np.isnan(gamma)] = 0

        return beta, gamma

    def computeLc1(self,h_xc,gamma,q):
        return np.abs((-h_xc - q*self.eps)/gamma[-1])

    def computeLc2(self,q):
        return np.abs(q*self.C - self.theta[-1])

    def computeLs(self,q,beta):
        """
        Compute the movement of all element in support set
        :param q:
        :param beta:
        :return:
        """
        support_theta = self.theta[self.supportSetIndices]
        support_beta = beta[self.supportSetIndices]
        s = self.sign_numpy(q*support_beta)
        s_s_theta = self.sign_numpy(s*support_theta)
        k_ = 1-(1-s_s_theta)/2
        Ls = (k_*s*self.C - support_theta)/support_beta
        Ls_min_index = np.argmin(np.abs(s))
        Ls_min_support_vector_index = self.supportSetIndices[Ls_min_index]
        return Ls, np.abs(Ls[Ls_min_index]), Ls_min_support_vector_index

    def computeLe(self,q,beta):
        error_set_feature = self.X[self.errorSetIndices]
        error_set_label = self.Y[self.errorSetIndices]
        H_e = self.computeMargin(error_set_feature, error_set_label)

        error_beta = beta[self.errorSetIndices]
        s = self.sign_numpy(q*error_beta)
        Le = (-H_e - s*self.eps)/error_beta

        Le_min_index = np.argmin(np.abs(Le))
        Le_min_support_vector_index = self.errorSetIndices[Le_min_index]
        return Le, np.abs(Le[Le_min_index]), Le_min_support_vector_index

    def computeLr(self,q,beta):
        remainder_set_feature = self.X[self.errorSetIndices]
        remainder_set_label = self.Y[self.errorSetIndices]
        H_e = self.computeMargin(remainder_set_feature, remainder_set_label)

        remaining_beta = beta[self.remainderSetIndices]
        s = self.sign_numpy(q * remaining_beta)
        Lr = (-H_e - s * self.eps) / remaining_beta

        Lr_min_index = np.argmin(np.abs(Lr))
        Lr_min_support_vector_index = self.errorSetIndices[Lr_min_index]
        return Lr, np.abs(Lr[Lr_min_index]), Lr_min_support_vector_index


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

    OSVR.save_model('online-svr.pkl')
    # Predict stuff as quick test
    testX = np.array([[0.15, 0.1], [0.25, 0.2]])
    testY = np.sin(2 * np.pi * testX)
    print('testX:', testX)
    print('testY:', testY)

    OSVR = OSVR.load_model('online-svr.pkl')
    PredictedY = OSVR.predict_(np.array([testX[0, :]]))
    Error = OSVR.computeMargin(testX[0], testY[0])
    print('PredictedY:', PredictedY)
    print('Error:', Error)
    return OSVR

if __name__ == '__main__':
    main(sys.argv)
    # print('run main plz')
