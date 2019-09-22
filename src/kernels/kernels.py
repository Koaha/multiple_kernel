import numpy as np
from scipy.spatial.distance import cdist
import scipy.sparse as sp
from src.utilities import instance_convert

class Linear():
    def __call__(self, a, b):
        x = np.array(a)
        y = np.array(b)
        return np.dot(x, y.T)

class Polynomial():
    def __call__(self,a,b,degree=2, gamma=None, coef0=1):
        x = np.array(a)
        y = np.array(b)
        if gamma == None:
            gamma = 1 / len(a)
        return (coef0 + gamma * x.dot(y.T)) ** degree

class RBF():
    def __init__(self,sigma=2.0,gamma=None):
        if gamma is None:
            self.gamma = 1.0 / (2.0 * sigma ** 2)
        else:
            self.gamma = gamma
    def __call__(self,x,y):
        if sp.isspmatrix(x) and sp.isspmatrix(y):
            x = np.array(x.todense())
            y = np.array(y.todense())
        if not hasattr(x, "shape"):
            return np.exp(-self.gamma * np.linalg.norm(x - y, ord=2) ** 2)
        if np.asarray(x).ndim == 0:
            return np.exp(-self.gamma * (x - y) ** 2)
        if len(x.shape) >= 2 or len(y.shape) >= 2:
            return np.exp(-self.gamma * cdist(x, y, metric="euclidean") ** 2)
        return np.exp(-self.gamma * np.linalg.norm(x - y, ord=2) ** 2)

class chi2_kernel():
    def __call__(self,X, Y, gamma=50.):
        """
        Chi^2 kernel,
        K(x, y) = exp( -gamma * SUM_i (x_i - y_i)^2 / (x_i + y_i) )
        https://lear.inrialpes.fr/pubs/2007/ZMLS07/ZhangMarszalekLazebnikSchmid-IJCV07-ClassificationStudy.pdf
        """
        kernel = np.zeros((X.shape[0], Y.shape[0]))

        for d in range(X.shape[1]):
            column_1 = X[:, d].reshape(-1, 1)
            column_2 = Y[:, d].reshape(-1, 1)
            kernel += (column_1 - column_2.T) ** 2 / (column_1 + column_2.T)
        return np.exp(gamma * kernel)

class min_kernel():
    def __call__(self,X, Y, param=None):
        """
        Min kernel (Histogram intersection kernel)
        K(x, y) = SUM_i min(x_i, y_i)
        http://perso.lcpc.fr/tarel.jean-philippe/publis/jpt-icip05.pdf
        """
        kernel = np.zeros((X.shape[0], Y.shape[0]))

        for d in range(X.shape[1]):
            column_1 = X[:, d].reshape(-1, 1)
            column_2 = Y[:, d].reshape(-1, 1)
            kernel += np.minimum(column_1, column_2.T)

        return kernel

class gen_min_kernel():
    def __call__(self,X, Y, alpha=1.):
        """
        Generalized histogram intersection kernel
        K(x, y) = SUM_i min(|x_i|^alpha, |y_i|^alpha)
        http://perso.lcpc.fr/tarel.jean-philippe/publis/jpt-icip05.pdf
        """
        mkernel = min_kernel()
        return mkernel(np.abs(X) ** alpha, np.abs(Y) ** alpha)

class euclidean_dist():
    def __call__(self,X, Y):
        """
        matrix of pairwise squared Euclidean distances
        """
        norms_1 = (X ** 2).sum(axis=1)
        norms_2 = (Y ** 2).sum(axis=1)
        return np.abs(norms_1.reshape(-1, 1) + norms_2 - 2 * np.dot(X, Y.T))

class laplacian_kernel():
    def __call__(self,X,Y, sigma=50):
        dists = euclidean_dist()
        return np.exp(-1 / sigma * np.sqrt(dists(X, Y)))

class simple_wavelet_kernel():
    def __call__(self,X, Y, param=1):
        val = (X-Y)/param
        mat = np.cos(1.75 * val) * np.exp(-val ** 2)
        # np.prod(mat, axis=1)
        return mat @ mat.T

class Custom():
    def set_arguments(self, X, fn, Y = None):
        self.X = X
        self.fn = fn
        self.Y = Y

    def __call__(self,a,b,**kwargs):
        operator = self.fn()
        X = instance_convert(self.X)
        if self.Y != None:
            Y = instance_convert(self.Y)
            # Y = self.Y()
            return operator(X(a,b),Y(a,b))
        return operator(X(a,b))