import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm
from src.kernels.kernels import RBF
from svm_helpers import (
    compute_kernel_output, compute_gradient, adjust_sets
)

class LaSVM(BaseEstimator, ClassifierMixin):
    """
    LaSVM is an online Support Vector Machine implementing the LaSVM 
    algorithm for large-scale, real-time SVM learning. It efficiently 
    maintains support vectors and dynamically adjusts based on incoming data.

    Parameters
    ----------
    C : float
        Regularization parameter.
    tau : float
        Threshold for violating pairs in the Sequential Minimal Optimization (SMO).
    kernel : callable, default=RBF()
        Kernel function to use; defaults to RBF.
    kernelParam : float, optional
        Parameter for the RBF kernel.
    eps : float, default=1e-4
        Tolerance for convergence.
    verbose : int, default=0
        Verbosity level for console output.
    epoc_offset : int, default=2000
        Offset for reprocessing frequency during training.

    Attributes
    ----------
    S : list of np.ndarray
        List of support vectors.
    a : list of float
        Coefficients for support vectors.
    g : list of float
        Gradient values for support vectors.
    y : list of int
        Labels of the support vectors.
    b : float
        Bias term.
    delta : float
        Difference in gradient values for convergence checks.
    """

    def __init__(self, C, tau, kernel=RBF(), kernelParam=None, eps=1e-4, verbose=0, epoc_offset=2000):
        self.C = C
        self.tau = tau
        self.kernel = kernel
        self.kernelParam = kernelParam
        self.eps = eps
        self.verbose = verbose
        self.epoc_offset = epoc_offset
        self.S, self.a, self.g, self.y = [], [], [], []
        self.b = 0
        self.delta = 0

    def lasvm_process(self, v, cls):
        """
        Processes a new sample for inclusion in the support vector set.

        Parameters
        ----------
        v : np.ndarray
            Feature vector of the new sample.
        cls : int
            Label of the new sample.
        """
        self.S.append(v)
        self.a.append(0)
        self.y.append(cls)
        self.g.append(cls - compute_gradient(self.S, np.array(self.a), self.b))
        if cls > 0:
            i = len(self.S) - 1
            _, j = self.extreme_ij()
        else:
            j = len(self.S) - 1
            i, _ = self.extreme_ij()
        if self.tau_violating(i, j):
            self.lbda(i, j)

    def lbda(self, i, j):
        """
        Updates coefficients for support vectors using lambda adjustment.

        Parameters
        ----------
        i : int
            Index of the first support vector.
        j : int
            Index of the second support vector.
        """
        S = np.array(self.S)
        kernel_ij = compute_kernel_output(self.kernel, S[i].reshape(1, -1), S[j].reshape(1, -1), self.kernelParam)
        lambda_val = min((self.g[i] - self.g[j]) / (2 * (1 - kernel_ij)),
                         self.C - self.a[i],
                         self.a[j] - 0)
        self.a[i] += lambda_val
        self.a[j] -= lambda_val

    def fit(self, X, Y):
        """
        Fits the LaSVM model to data using an online learning approach.

        Parameters
        ----------
        X : np.ndarray
            Training data, shape (n_samples, n_features).
        Y : np.ndarray
            Labels for training data, shape (n_samples,).
        """
        X, Y = np.array(X), np.array(Y)
        Y[Y == 0] = -1
        self.init_samples(X, Y)
        for i in tqdm(range(X.shape[0])):
            self.update(X[i, :], Y[i])
            if i % self.epoc_offset == 0 or i == (X.shape[0] - 1):
                while self.delta > self.tau:
                    self.lasvm_reprocess()

    def predict(self, X):
        """
        Predicts binary class labels for each sample in X.

        Parameters
        ----------
        X : np.ndarray
            Data to predict, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted class labels, shape (n_samples,).
        """
        results = []
        for v in X:
            prediction = compute_gradient(self.S, np.array(self.a), self.b) + self.b
            results.append(1 if prediction > 0 else 0)
        return np.array(results)

    def lasvm_reprocess(self):
        """
        Reprocesses support vectors to ensure convergence.
        """
        i, j = self.extreme_ij()
        while self.tau_violating(i, j):
            self.lbda(i, j)
            i, j = self.extreme_ij()
