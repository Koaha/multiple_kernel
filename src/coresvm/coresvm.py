import numpy as np
import cvxpy as cp
import pickle
from math import sqrt
from random import uniform, shuffle,seed, choice, randint
import sys,os
import pandas as pd
sys.path.append("../../")
sys.path.append("../")
# sys.path.append(".")
from kernels.kernels import RBF,Linear,Polynomial

class CoreSVM():

    def __init__(self,X,Y,kernel=None,C = 100,iter = None):
        #TODO check self.delta
        self.row,self.col = np.shape(X)
        self.X = X
        self.Y = Y
        self.C = C
        self.Y[self.Y == 0] = -1 # some stupid convention 0 -> -1
        self.w = np.zeros((self.col,1))
        self.alpha = np.zeros((self.row,1))
        self.delta = 0.5
        if iter != None:
            self.iter = iter
        else:
            self.iter = -1

        if kernel ==None:
            self.kernel = RBF()
        else:
            self.kernel = kernel
        return

    def compute_kernel_prime(self,xi,yi,xj,yj):
        return yi * yj * self.kernel(xi, xj) + yi * yj + self.delta / self.C

    # Step 4.1 Initialization
    def initialize_set(self):
        """
        :compute:
            S0: the set of center points
            c0: the center of the ball w.r.t the projection space
            - init the Langrange coefficent instead
            R0: the ball radius
        """
        S0 = np.array([])
        pos_samples = X[Y == 1]
        neg_samples = X[Y == -1]
        y_pos_samples = X[Y == 1]
        y_neg_samples = X[Y == -1]

        #get arbitrary point from pos set and neg set
        pos_index = np.random.randint(len(pos_samples))
        neg_index = np.random.randint(len(neg_samples))
        za = pos_samples[pos_index]
        ya = y_pos_samples[pos_index]
        yb = y_neg_samples[neg_index]
        zb = neg_samples[neg_index]
        S0 = np.append(S0,za)
        S0 = np.append(S0, za)

        # the coefficient correspond to the center point
        # c = 0.5(omega(x_pos) + omega(x_neg))
        self.alpha[pos_index] = 0.5
        self.alpha[neg_index] = 0.5

        # omega_prime_za = ya ** 2 * self.kernel(za, za) + ya ** 2 + 1 / self.C
        omega_prime_za = self.compute_kernel_prime(za,ya,za,ya)
        omega_prime_zb = self.compute_kernel_prime(zb,yb,zb,yb)
        omega_prime_za_zb = self.compute_kernel_prime(za,ya,zb,yb)
        omega_prime_zb = yb ** 2 * self.kernel(zb, zb) + yb ** 2 + 1 / self.C
        omega_prime_za_zb = yb * yb * self.kernel(za, zb) + yb * yb + self.delta/self.C
        R0 = 0.5*np.sqrt(omega_prime_za+omega_prime_zb-2*omega_prime_za_zb)
        return S0,R0

    # Step 4.2 Distance computations
    def compute_distance(self,X,Y,xl,yl):
        """

        :param X: X of Z in the enclosing set at the t iteration
        :param Y: Y of Z in the enclosing set at the t iteration
        :param xl: the current checking vector
        :param yl:
        :return:
        """
        S_set_distance = self.alpha.T * self.alpha *self.compute_kernel_prime(self.X,self.Y,self.X,self.Y)
        S_set_distance_with_l = self.alpha * \
                                self.compute_kernel_prime(X,Y,np.tile(xl,(len(xl),1)),np.tile(yl,(len(yl),1)))
        K_l_l = self.compute_kernel_prime(xl,yl,xl,yl)
        return np.sum(S_set_distance) - 2*np.sum(S_set_distance_with_l) + K_l_l

    #Step 4.3 Adding the furthest point
    def adding_furthest_point(self):
        return

if  __name__ == "__main__":
    DATA_PATH = os.path.join(os.getcwd(),"../../data")
    X = np.array(pd.read_csv(os.path.join(DATA_PATH,"toyX.csv")))
    Y = np.array(pd.read_csv(os.path.join(DATA_PATH, "toyY.csv")))
    Y[Y==0] = -1

    kernel = RBF(gamma=50)
    cvm = CoreSVM(X,Y,iter=2000,kernel=kernel)


