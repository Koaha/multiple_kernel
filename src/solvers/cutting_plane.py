import numpy as np
import cvxpy as cp
class CP():

    """
       :param X: matrix with n rows (data points) and d columns (features)
       :param Y: the n-row label vector
       :param C: Trade-off control paramters
       :param epsilon: tolerance rate
       :type X: 2D-numpy
       :type y: 1D-numpy
       :type C: float
       :type epsilon: float
       :return w, xi_prime:
          * w: separation plane
          * sensitivity error eta
       :rtype: array

       :Example:
            >
            >
            >
    """
    def CP_primal(self, X, Y, C, epsilon):
        #initilize parameters
        if C == None:
            C = 100
        n_dim = 2
        n_row = 5
        size_to_train = 5
        delta_prime = np.zeros((size_to_train, 1))
        psi_prime = np.zeros((size_to_train, 1))
        w = cp.Variable((n_row, 1))
        si = cp.Variable((size_to_train))
        obj = cp.Minimize(1 / 2 * w.T * w + C/size_to_train * cp.sum(si))
        constraints = []
        for i in range(size_to_train):
            constraints.append(w.T @ psi_prime[i] >= delta_prime[i] - si[i])
            constraints.append(si[i]>=0)
        prob = cp.Problem(obj,constraints)
        w_out,si_out = prob.solve()






        return

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

    def CP_dual(self, X, s, kernelFunc, accelerated_flag=0):
        # initilize parameters

        return

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

    def CPSP(self, X, s, kernelFunc, accelerated_flag=0):
        # initilize parameters

        return