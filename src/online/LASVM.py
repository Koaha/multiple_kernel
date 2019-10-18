
# coding: utf-8
 
"""Online learning."""
import timeit,time,sys
import pandas as pd

from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
sys.path.append("..")
from kernels.kernel import RBF,Linear
from utilities.gram import gram
from utilities import *

class LaSVM(BaseEstimator,ClassifierMixin):
    def __init__(self, C, tau, kernel=RBF(), kernelParam=None, eps=1e-4,verbose=0,epoc_offset=2000):
        self.S = []    # Set S contains potential support vectors
        self.a = []    # Coefficent of current kernel
        self.g = []    # partial derivatives
        self.y = []
        self.C = C
        self.kernelParam = kernelParam
        self.verbose = verbose
        self.k = kernel
        self.epoc_offset = epoc_offset
        self.tau = tau
        self.eps = eps
        self.b = 0
        self.delta = 0
        self.i = 0
        self.misses = 0

    # Compute  output kernels K(x_j,x) with size (n x m)
    # n: number of support vectors in support set
    # m: number of samples for prediction
    def compute_K(self, support_set, input_sample):
        if (self.k.__class__.__name__ == 'RBF'):
            K = self.k.__call__(support_set, input_sample, gamma=self.kernelParam)
        else:
            K = gram()
            K = K.__call__(support_set, input_sample, self.k)
        return K

    def compute_g(self, v):
        alpha = np.array(self.a).reshape(-1,1).tolist()
        res = (np.array(self.compute_K(np.array(v).reshape(1, -1), np.array(self.S))).dot(alpha)[0])
        return np.asscalar(res)

    def predict(self, X):
        res = []
        for singleX in X:
            pred_value = (self.compute_g(singleX)) + self.b
            res.append(pred_value)
        res = np.array(res).reshape(-1,1)
        res[res > 0] = 1
        res[res <= 0] = 0
        return res.astype(int)

    def A(self, i):
        return min(0, self.C*self.y[i])
    
    def B(self, i):
        return max(0, self.C*self.y[i])

    def tau_violating(self, i, j):
        return ((self.a[i] < self.B(i)) and
                (self.a[j] > self.A(j)) and
                ((self.g[i] - self.g[j]) > self.tau))

    def extreme_ij(self):
        S = self.S
        i = np.argmax(list((self.g[i] if self.a[i]<self.B(i) else -np.inf)
                           for i in range(len(S))))
        j = np.argmin(list((self.g[i] if self.a[i]>self.A(i) else np.inf)
                           for i in range(len(S))))
        return i,j

    def lbda(self, i, j):
        S = self.S
        l= min((self.g[i]-self.g[j])/(self.compute_K(S[i].reshape(1,-1),S[i].reshape(1,-1))
                                      +self.compute_K(S[j].reshape(1,-1),S[j].reshape(1,-1))
                                      -2*self.compute_K(S[i].reshape(1,-1),S[j].reshape(1,-1))),
               self.B(i)-self.a[i],
               self.a[j]-self.A(j))
        self.a[i] += l
        self.a[j] -= l
        for s in range(len(S)):
            self.g[s] -= l*(self.compute_K(S[i].reshape(1,-1),S[s].reshape(1,-1))
                            -self.compute_K(S[j].reshape(1,-1),S[s].reshape(1,-1)))
        return l
    
    def lasvm_process(self, v, cls, w=0):
        self.S.append(v)
        self.a.append(0)
        self.y.append(cls)
        self.g.append(cls - self.compute_g(v))
        if cls > 0:
            i = len(self.S)-1
            foo, j = self.extreme_ij()
        else:
            j = len(self.S)-1
            i, foo = self.extreme_ij()
        if not self.tau_violating(i, j):
            return
        return self.lbda(i,j)

    def lasvm_reprocess(self):
        #first search a pair of tau violating
        # with maximal gradient
        S = self.S
        i,j = self.extreme_ij()
        if not self.tau_violating(i,j):
            return
        self.lbda(i,j)  #update i,j after compute lambda
        i,j = self.extreme_ij()
        to_remove = []

        for s in range(len(S)):
            if  self.a[s] == 0:
                if self.y[s] <= 0 and self.g[s]>=self.g[i]:
                    to_remove.append(s)
                elif self.y[s] > 0 and self.g[s]<=self.g[j]:
                    to_remove.append(s)

        # remove remain sets
        for s in reversed(to_remove):
            del S[s]
            del self.a[s]
            del self.y[s]
            del self.g[s]
        i,j = self.extreme_ij()
        self.b = (self.g[i]+self.g[j])/2.
        self.delta = self.g[i]-self.g[j]

    def add_sample(self,v,c):
        self.S.append(v)
        self.y.append(c)
        self.a.append(0)
        self.g.append(c - self.compute_g(v)+self.b)

    # Feed model with some support vectors from each class
    def init_sample(self,X,Y):
        X = np.array(X)
        Y = np.array(Y)
        for i in tqdm(range(X.shape[0])):
            check_list = np.array(self.y)
            if len(check_list[check_list > 0]) <10 and Y[i]>0:
                self.add_sample(X[i, :],Y[i])
            elif len(check_list[check_list <= 0]) < 10 and Y[i]<=0:
                self.add_sample(X[i, :], Y[i])
            elif len(check_list[check_list > 0]) >= 10 and len(check_list[check_list <= 0]) >= 10:
                return

    def update(self, v, c):
        if c*(self.compute_g(v) + self.b) < 0:
            self.misses += 1
        self.i += 1
        self.lasvm_process(v,c)
        self.lasvm_reprocess()
        if self.i % 1000 == 0:
            print("m", self.misses, "s", len(self.S))
            self.misses = 0

    def fit(self,X,Y):
        samples = []
        run_time = []
        start = time.time()
        X = np.array(X)
        Y = np.array(Y)
        Y[Y==0] = -1
        self.init_sample(X,Y)

        for i in tqdm(range(X.shape[0])):
            self.update(X[i, :],Y[i])
            if self.verbose == 1:
                print('%%%%%%%%%%%%%%% Data point {0} %%%%%%%%%%%%%%%'.format(i))
            if i == (X.shape[0]-1) or (i%self.epoc_offset == 0):
                while self.delta > self.tau:
                    self.lasvm_reprocess()
            stop = time.time()
            samples.append(i)
            run_time.append(stop-start)
            # print(i,' sample; duration = ', stop-start)
        df = pd.DataFrame({'samples':samples,'run_time':run_time})
        df.to_csv(self.__class__.__name__+"_runtime.csv")