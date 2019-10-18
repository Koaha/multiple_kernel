import numpy as np
import cvxpy as cp
import pickle
from math import sqrt
from random import uniform, shuffle,seed, choice, randint
import sys,os
import pandas as pd
sys.path.append("../../")
sys.path.append("../")
from kernels.kernels import RBF,Linear,Polynomial

class Pegasos():

    def __init__(self,X,Y,kernel=None,lamb=1,iter = 2000):
        self.row,self.col = np.shape(X)
        self.X = X
        self.Y = self.sign(Y)
        self.lamb = lamb
        self.w = np.zeros((self.col,1))
        self.alpha = np.zeros((self.row,1))
        self.iter = iter

        if kernel ==None:
            self.kernel = RBF()
        else:
            self.kernel = kernel
        return

    def sign(self,inp):
        out = np.sign(inp)
        out[out==0] = 1
        return out

    # ================================
    # START OF PEGASOS ON SINGLE ENTRY
    # ================================
    def hinge_loss(self,i):
        return max(0,1-self.Y[i].T*(self.X[i]@self.w))

    def obj_fn(self,i):
        self.lamb/2*np.linalg.norm(self.w)**2 + self.hinge_loss(i)

    def predict(self,inp):
        return (inp @ self.w)

    #compute subgradient or delta whatsoever
    def compute_subgradient(self,i):
        return self.lamb*self.w - self.indicator_function(i)*(self.sign(self.Y[i]) * self.X[i]).reshape(-1,1)

    def indicator_function(self,i):
        if self.sign(self.Y[i]) * self.predict(X[i]) <1:
            return 1
        return 0

    def solve_single(self):
        # random select  a training example.
        # in other term, take sub
        for t in range(1,self.iter):
            i = np.random.randint(self.row)
            delta_t = self.compute_subgradient(i)
            step_size = 1/(self.lamb * t)
            self.w = self.w - step_size*delta_t
        return self.w
    # ==============================
    # END OF PEGASOS ON SINGLE ENTRY
    # ==============================

    def check_alpha_to_incre(self,i,t):
        K = self.kernel(self.X[i].reshape(1,-1),self.X)
        #compute hadamard product, element wise product between alpha set and y
        alpha_y = np.multiply(self.sign(self.Y),self.alpha)
        if (self.sign(self.Y[i])/(self.lamb *t)*(K@alpha_y))<1:
            return True
        return False

    def update_alpha_j(self,i,t):
        """
        :param t: the t iteration for measuring step size
        :param i: the current random sample
        :return: alpha at the t iteration
        """
        if self.check_alpha_to_incre(i,t):
            self.alpha[i] = self.alpha[i]+1
        return

    def solve_kernel(self):
        # random select  a training example.
        # in other term, take sub
        for t in range(1, self.iter):
            i = np.random.randint(self.row)
            self.update_alpha_j(i,t)
        return self.alpha

    def predict_kernel(self,inp,t=None):
        if t == None:
            t = self.iter
        step_size = 1/(self.lamb*t)
        K = self.kernel(self.X,inp)
        alpha_y = np.multiply(self.sign(self.Y),self.alpha)
        return step_size*(alpha_y.T@K)

if  __name__ == "__main__":
    DATA_PATH = os.path.join(os.getcwd(),"../../data")
    X = np.array(pd.read_csv(os.path.join(DATA_PATH,"toyX.csv")))
    Y = np.array(pd.read_csv(os.path.join(DATA_PATH, "toyY.csv")))
    Y[Y==0] = -1

    kernel = RBF(gamma=50)
    primal = Pegasos(X,Y,iter=2000,kernel=kernel)
    #==================================
    #START test pegasos on single learn
    # ==================================
    # primal.solve_single()
    # print(primal.w)
    # print(primal.predict(X))
    # ==================================
    # END test pegasos on single learn
    # ==================================

    #==================================
    #START test pegasos with kernel
    # ==================================
    primal.solve_kernel()
    print(primal.alpha)
    print(primal.predict_kernel(X))
    # ==================================
    # END test pegasos with kernel
    # ==================================

