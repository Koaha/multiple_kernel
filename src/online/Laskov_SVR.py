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
from kernels.RBF import RBF
from utilities.gram import gram
from utilities import *
from copy import copy
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
import time

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
        self.weights = np.array([])

        # Samples X (features) and Y (truths)
        self.X = list()
        self.Y = list()
        # Working sets, contains indices pertaining to X and Y
        self.supportSetIndices = list()
        self.errorSetIndices = list()
        self.remainderSetIndices = list()
        self.R = np.matrix([])

        if kernel == None:
            self.kernel = RBF()
        else:
            self.kernel = kernel

    def learn(self, newSampleX, newSampleY):
        #Line 1 Add X,Y to the training set
        self.numSamplesTrained +=1 #increase the sample size
        self.X.append(newSampleX)# add new instance to the data flow
        self.Y.append(newSampleY)
        self.weights = np.append(self.weights, 0)
        i = self.numSamplesTrained - 1
        # The new set with additional new Sample input
        H = self.computeMargin(self.X,self.Y)
        #H(xc) is the Margin at index xc = i (latest sample)
        if (abs(H[i]) <= self.eps):
            # Line 3 add new sample to the remaining set and exit
            if self.verbose == 1:
                print('Adding new sample {0} to remainder set, within eps.'.format(i))
            self.remainderSetIndices.append(i)
            return

        addNewSample = False
        iterations = 0
        # Step 6 assign new sample to either 1 of the 3 groups
        # Support, Error, Remaining
        while not addNewSample:
            # print(len(self.supportSetIndices),'self.supportSetIndices =', self.supportSetIndices)
            # print(len(self.remainderSetIndices),'self.remainderSetIndices =', self.remainderSetIndices)
            # print(len(self.errorSetIndices),'self.errorSetIndices =', self.errorSetIndices)
            # Ensure we're not looping infinitely
            iterations += 1
            if iterations > self.numSamplesTrained * 100:
                print('Warning: we appear to be in an infinite loop.')
                sys.exit()
                iterations = 0
            # Line6.1 Update beta gamma at the new sample i
            # Using equation 10 & 12
            beta,gamma = self.computeBetaGamma(i)
            #Line6.2 Find least variations

            deltaC, flag, minIndex = getMinVariation(self,H, beta, gamma, i)
            # Update weights and bias based on variation
            if len(self.supportSetIndices) > 0 and len(beta) > 0:
                self.weights[i] += deltaC
                delta = beta * deltaC
                self.bias += delta.item(0)
                weightDelta = np.array(delta[1:])
                weightDelta.shape = (len(weightDelta),)
                self.weights[self.supportSetIndices] += weightDelta
                H += gamma * deltaC
            else:
                self.bias += deltaC
                H += deltaC
            # Adjust sets, moving samples between them according to flag
            H, addNewSample = self.adjustSets(H, beta, gamma, i, flag, minIndex)
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


    def prune_vector(self,kn=7):
        # TODO this part does 2 tasks
        if self.prune:
            i = self.numSamplesTrained - 1
            H = self.computeMargin(self.X, self.Y)
            beta, gamma = self.computeBetaGamma(i)
            get_reduce_list = getNMinSupport(self, H, beta, gamma, i, kn=kn)
            delta = self.move_less(get_reduce_list)
            if len(get_reduce_list) > 0:
                self.balance_equilibrium(delta, get_reduce_list)
        if self.verbose == 1:
            print(len(self.remainderSetIndices), 'self.remainderSetIndices =', self.remainderSetIndices)
            print(len(self.errorSetIndices), 'self.errorSetIndices =', self.errorSetIndices)
        return

    def get_less_significant_list(self,kn=4):
        if len(self.supportSetIndices) <= (kn+1):
            return []

        number_less_significant = int (len(self.supportSetIndices)/(kn+1))
        less_significant_list = np.array(abs(self.weights)).argsort()[:number_less_significant][::-1]
        delta = self.weights[less_significant_list]
        less_significant_list = [item for item in less_significant_list if item not in self.remainderSetIndices
                                 and item not in self.errorSetIndices]
        # less_significant_list = np.delete(less_significant_list,
        #                                   np.where(less_significant_list ==self.remainderSetIndices))
        # less_significant_list = np.delete(less_significant_list,
        #                                   np.where(less_significant_list == self.errorSetIndices))
        return less_significant_list

    # def move_less(self,less_significant_list):
    #     i = self.numSamplesTrained - 1
    #     H = self.computeMargin(self.X, self.Y)
    #     delta = -self.weights[less_significant_list]
    #     if len(less_significant_list) <1:
    #         return self.computeMargin(self.X, self.Y)-H
    #     beta, gamma = self.computeBetaGamma(i)
    #     # set flag to move to
    #
    #     for index in less_significant_list:
    #         if index in self.supportSetIndices:
    #             minIndex = self.supportSetIndices.index(index)
    #             weightsValue = self.weights[index]
    #             if np.abs(weightsValue) < np.abs(self.C - abs(weightsValue)):
    #                 self.weights[index] = 0
    #                 weightsValue = 0
    #             else:
    #                 self.weights[index] = np.sign(weightsValue) * self.C
    #                 weightsValue = self.weights[index]
    #             # Move from support to remainder set
    #             if weightsValue == 0:
    #                 if self.verbose == 1:
    #                     print('Moving sample {0} from support to remainder set.'.format(index))
    #                 self.remainderSetIndices.append(index)
    #                 self.R = self.removeSampleFromR(minIndex)
    #                 self.supportSetIndices.pop(minIndex)
    #             # move from support to error set
    #             elif np.abs(weightsValue) == self.C:
    #                 if self.verbose == 1:
    #                     print('Moving sample {0} from support to error set.'.format(index))
    #                 self.errorSetIndices.append(index)
    #                 self.R = self.removeSampleFromR(minIndex)
    #                 self.supportSetIndices.pop(minIndex)
    #             else:
    #                 if self.verbose == 1:
    #                     print('Issue with set swapping, flag 2.', 'weightsValue:', weightsValue)
    #                 sys.exit()
    #     # deltaH = -self.computeMargin(self.X, self.Y)
    #     return delta

    def balance_equilibrium(self,deltaH,least_sig_list):
        R = np.matrix(self.R).copy()
        X = np.matrix(self.X).copy()
        SR = self.computeQ(X[self.supportSetIndices,:],X[least_sig_list])
        SR = np.vstack((np.ones(len(least_sig_list)),SR))
        SS = (R@SR)@deltaH
        self.bias = self.bias + SS[0,0]
        self.weights[self.supportSetIndices] = self.weights[self.supportSetIndices]+SS[0,1:]
        # if (self.weights[abs(self.weights)>self.C]):
        #     print('violate constraints')
        if self.verbose == 1:
            print('SS',SS)
        return

    def adjustSets(self, H, beta, gamma, i, flag, minIndex):
        if self.verbose == 1:
            print('Entered adjustSet logic with flag {0} and minIndex {1}.'.format(flag, minIndex))
        if flag not in range(5):
            print('Received unexpected flag {0}, exiting.'.format(flag))
            sys.exit()
        # add new sample to Support set
        if flag == 0:
            if self.verbose == 1:
                print('Adding new sample {0} to support set.'.format(i))
            H[i] = np.sign(H[i]) * self.eps
            self.supportSetIndices.append(i)
            self.R = self.addSampleToR(i, 'SupportSet', beta, gamma)
            return H, True
        # add new sample to Error set
        elif flag == 1:
            if self.verbose == 1:
                print('Adding new sample {0} to error set.'.format(i))
            self.weights[i] = np.sign(self.weights[i]) * self.C
            self.errorSetIndices.append(i)
            return H, True
        # move sample from Support set to Error or Remainder set
        elif flag == 2:
            index = self.supportSetIndices[minIndex]
            weightsValue = self.weights[index]
            if np.abs(weightsValue) < np.abs(self.C - abs(weightsValue)):
                self.weights[index] = 0
                weightsValue = 0
            else:
                self.weights[index] = np.sign(weightsValue) * self.C
                weightsValue = self.weights[index]
            # Move from support to remainder set
            if weightsValue == 0:
                if self.verbose == 1:
                    print('Moving sample {0} from support to remainder set.'.format(index))
                self.remainderSetIndices.append(index)
                self.R = self.removeSampleFromR(minIndex)
                self.supportSetIndices.pop(minIndex)
            # move from support to error set
            elif np.abs(weightsValue) == self.C:
                if self.verbose == 1:
                    print('Moving sample {0} from support to error set.'.format(index))
                self.errorSetIndices.append(index)
                self.R = self.removeSampleFromR(minIndex)
                self.supportSetIndices.pop(minIndex)
            else:
                if self.verbose == 1:
                    print('Issue with set swapping, flag 2.', 'weightsValue:', weightsValue)
                sys.exit()
            # print('Ignore moving sample from support to error set')
            # return H, True
        # move sample from Error set to Support set
        elif flag == 3:
            index = self.errorSetIndices[minIndex]
            if self.verbose == 1:
                print('Moving sample {0} from error to support set.'.format(index))
            H[index] = np.sign(H[index]) * self.eps
            self.supportSetIndices.append(index)
            self.errorSetIndices.pop(minIndex)
            self.R = self.addSampleToR(index, 'ErrorSet', beta, gamma)
        # move sample from Remainder set to Support set
        elif flag == 4:
            index = self.remainderSetIndices[minIndex]
            if self.verbose == 1:
                print('Moving sample {0} from remainder to support set.'.format(index))
            H[index] = np.sign(H[index]) * self.eps
            self.supportSetIndices.append(index)
            self.remainderSetIndices.pop(minIndex)
            self.R = self.addSampleToR(index, 'RemainingSet', beta, gamma)
        return H, False

    def addSampleToR(self, sampleIndex, sampleOldSet, beta, gamma):
        if self.verbose == 1:
            print('Adding sample {0} to R matrix.'.format(sampleIndex))
        X = np.array(self.X)
        sampleX = X[sampleIndex, :]
        sampleX.shape = ((int)(sampleX.size / self.numFeatures), self.numFeatures)
        # Add first element
        if self.R.shape[0] <= 1:
            Rnew = np.ones([2, 2])
            Rnew[0, 0] = -self.computeKernelOutput(sampleX, sampleX)
            Rnew[1, 1] = 0
        # Other elements
        else:
            # recompute beta/gamma if from error/remaining set
            if sampleOldSet == 'ErrorSet' or sampleOldSet == 'RemainingSet':
                # beta, gamma = self.computeBetaGamma(sampleIndex)
                Qii = self.computeKernelOutput(sampleX, sampleX)
                Qsi = self.computeKernelOutput(X[self.supportSetIndices[0:-1], :], sampleX)
                beta = -self.R @ np.append(np.matrix([1]), Qsi, axis=0)
                beta[np.isnan(beta)] = 0
                beta.shape = (len(beta), 1)
                gamma[sampleIndex] = Qii + np.append(1, Qsi.T) @ beta
                gamma[np.isnan(gamma)] = 0
                gamma.shape = (len(gamma), 1)
            # add a column and row of zeros onto right/bottom of R
            r, c = self.R.shape
            Rnew = np.append(self.R, np.zeros([r, 1]), axis=1)
            Rnew = np.append(Rnew, np.zeros([1, c + 1]), axis=0)
            # update R
            if gamma[sampleIndex] != 0:
                # Numpy so wonky! SO WONKY.
                beta1 = np.append(beta, [[1]], axis=0)
                Rnew = Rnew + 1 / gamma[sampleIndex].item() * beta1 @ beta1.T
            if np.any(np.isnan(Rnew)):
                print('R has become inconsistent. Training failed at sampleIndex {0}'.format(sampleIndex))
                sys.exit()
        return Rnew

    def removeSampleFromR(self, sampleIndex):
        if self.verbose == 1:
            print('Removing sample {0} from R matrix.'.format(sampleIndex))
        sampleIndex += 1
        I = list(range(sampleIndex))
        I.extend(range(sampleIndex + 1, self.R.shape[0]))
        I = np.array(I)
        I.shape = (1, I.size)
        if self.debug:
            print('I', I)
            print('RII', self.R[I.T, I])
        # Adjust R
        if self.R[sampleIndex, sampleIndex] != 0:
            Rnew = self.R[I.T, I] - (self.R[I.T, sampleIndex] * self.R[sampleIndex, I]) / self.R[
                sampleIndex, sampleIndex].item()
        else:
            Rnew = np.copy(self.R[I.T, I])
        # Check for bad things
        if np.any(np.isnan(Rnew)):
            print('R has become inconsistent. Training failed removing sampleIndex {0}'.format(sampleIndex))
            sys.exit()
        if Rnew.size == 1:
            print('Time to annhilate R? R:', Rnew)
            Rnew = np.matrix([])
        return Rnew

    # target new sample c in the paper equation is the index i
    # Q_ij = yi.yj.K(xi,xj)
    # Q is the margin vector working set
    def computeBetaGamma(self,i):
        # Compute the kernel of support set with
        # the new sample vector c (or i in this code) : Qsi
        X = np.matrix(self.X).copy()
        Qsi = self.computeQ(X[self.supportSetIndices,:],X[i,:])
        if len(self.supportSetIndices) == 0 or self.R.size == 0:
            beta = np.array([])
        else:
            beta = -self.R @ np.append(np.matrix([1]),Qsi,axis=0)

        Qxi = self.computeQ(X, X[i, :]) #compute kernel of all samples vs new samples
        Qxs = self.computeQ(X, X[self.supportSetIndices, :]) # compute kernel of all samples vs support set
        if len(self.supportSetIndices) == 0 or Qxi.size == 0 or Qxs.size == 0 or beta.size == 0:
            gamma = np.array(np.ones_like(Qxi))
        else:
            gamma = Qxi + np.append(np.ones([self.numSamplesTrained, 1]), Qxs, 1) @ beta

        # Correct for NaN
        beta[np.isnan(beta)] = 0
        gamma[np.isnan(gamma)] = 0
        return beta, gamma

    def computeQ(self,support_set,sample_set):
        Q = np.matrix(np.zeros([support_set.shape[0], sample_set.shape[0]]))
        if support_set.shape[0] > 0 and sample_set.shape[0] > 0:
            Q = self.computeKernelOutput(np.matrix(sample_set),
                                         np.matrix(support_set)).T
        return np.matrix(Q)

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
    def predict_(self, newSampleX):
        if (self.numSamplesTrained>0):
            K = self.computeKernelOutput(np.matrix(self.X).copy(),np.matrix(newSampleX).copy())
            return K.T.dot(self.weights.reshape(-1, 1)) + self.bias
        else:
            return np.zeros_like(newSampleX) + self.bias

    def pred(self, newSampleX):
        X = np.array(self.X)
        newX = np.array(newSampleX)
        weights = np.array(self.weights)
        weights.shape = (weights.size, 1)
        # if self.numSamplesTrained > 0:
        #     y = self.computeKernelOutput(X, newX)
        #     return np.sign((weights.T @ y).T + self.bias - 0.5)
        # else:
        #     return np.sign(np.zeros_like(newX) + self.bias - 0.5)
        if self.numSamplesTrained > 0:
            y = self.computeKernelOutput(X, newX)
            return np.sign((weights.T @ y).T + self.bias)
        else:
            return np.sign(np.zeros_like(newX) + self.bias)

    def predict(self, X):
        res = []
        for singleX in X:
            pred_value = self.pred([singleX])[0,0]
            res.append(pred_value)
        res = np.array(res)
        res[res > 0] = 1
        res[res <= 0] = 0
        return res.astype(int)

    # Line 3 compute the margin H(x) by taking
    # the differences with the prediction F(x)
    def computeMargin(self, newSampleX, newSampleY):
        fx = self.predict_(newSampleX)  # input is the set X + new instance X
        newSampleY = np.array(newSampleY)
        return fx - newSampleY



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
