import numpy as np
from math import sqrt
from random import uniform, shuffle,seed, choice, randint
sys.path.append("..")
from kernels.kernels import RBF,Linear

class nystorm():

    def ridge_leverage_score(self, K , lamb,debug = False):
        if debug:
            K = np.array([[1, 2, 3, 4, 5],
                          [2, 6, 7, 8, 9],
                          [3, 7, 10, 11, 12],
                          [4, 8, 11, 13, 14],
                          [5, 9, 12, 14, 15]])
        r,c = K.shape
        l_K = np.linalg.inv(K + lamb * np.eye(r, c))
        return np.diag(l_K)

    def rls_nystrom_sampling(self,X,kernelFunc,lamb,delta):
        K_out = np.zeros()
        return K_out

    """
       This file implements Algorithm 3 as described in
       https://arxiv.org/abs/1605.07583

       :param X: matrix with n rows (data points) and d columns (features)
       :param s: the number of samples used to construct the Nystrom
        approximation. default = sqrt(n). Generally should set s < n.
       :param kernelFunc: A function that can compute arbitrary submatrices of
      X's kernel matrix for some positive semidefinite kernel. For
      implementation specifics, see the provided example gaussianKernel.m
       :param accelerated_flag: either 0 or 1. default = 0. If the flag is set to 1,
      the code uses an accelerated version of the algorithm as described
      in Section 5.2.1 of https://arxiv.org/abs/1605.07583. This version
      will output a lower quality Nystrom approximation, but will run more
      quickly. We recommend setting accelerated_flag = 0 (the default)
      unless the standard version of the algorithm runs too slowly for
      your purposes
       :type X: 2D-numpy
       :type s: int
       :type kernelFunc: kernel class
       :type accelerated_flag: int [0-1]
       :return [C, W]: * C : A subset of s columns from A's n x n kernel matrix.
          * W : An s x s positive semidefinite matrix such that
          C*W*C' approximates K.
       :rtype: array

       :Example:
            %  Compute a Nystrom approximation for a Gaussian kernel matrix with
            %  variance parameter gamma = 40,. I.e. the kernel function for points
            %  x,y is e^-(40*||x - y||^2).
            %
            %  gamma = 40;
            %  kFunc = @(X,rowInd,colInd) gaussianKernel(X,rowInd,colInd,gamma);
            %  [C,W] = recursiveNystrom(X,s,kFunc);
    """
    def recursiveNystorm(self, X,s,kernelFunc,accelerated_flag = 0):
        #initilize parameters

        if s == None:
            s = np.ceil(np.sqrt(X.shape[0]))    #To be checked
        n,d = X.shape
        sLevel = np.ceil(np.sqrt((n*s + s^3)/(4*n)))
        # other  case if accelerated  is not defined then return sLevel = s

        #start the approximation
        oversamp = np.log(sLevel)
        k = np.ceil(sLevel/(4*oversamp))
        nLevels = np.ceil(np.log(n/sLevel)/np.log(2))

        # random permutation for uniform input samples
        perm = np.random.permutation(n)

        # set up size of recursive levels
        lSize = np.zeros(nLevels+1)
        lSize[0] = n;

        for i in range(1,nLevels+1):
            lSize[i] = np.ceil(lSize[i-1]/2)
        lSize = lSize.reshape(1, len(lSize))

        # rInd: indices of points selected at previous level of recursion
        # at the base level it's just a uniform sample of ~sLevel points
        samp = np.arange(lSize[-1])
        rInd = perm[samp]
        weights = np.ones(len(rInd))

if __name__ == "__main__":
    A = np.random.rand(10000, 10)
    ny = nystorm()
    rbf = RBF()
    linear = Linear()

    nystrom_output = ny.recursiveNystorm(A,100,rbf)
    kernel_output = rbf(A, A)

    ker_diff = np.abs(nystrom_output - kernel_output)
    print(ker_diff)
    print(sum(ker_diff))