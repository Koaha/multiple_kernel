import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm
from src.kernels.kernels import RBF
from src.metrics.metrics_tracker import (
    MetricsTracker,
)  # Ensure MetricsTracker is imported correctly
from svm_helpers import compute_kernel_output, compute_gradient, adjust_sets


class LaSVM(BaseEstimator, ClassifierMixin):
    """
    LaSVM is an online Support Vector Machine implementing the LaSVM
    algorithm for large-scale, real-time SVM learning. It efficiently
    maintains support vectors and dynamically adjusts based on incoming data.
    Integrates real-time metric tracking for online performance evaluation.

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
    enable_metrics : bool, optional, default=True
        Enables real-time metrics tracking during training if set to True.

    Attributes
    ----------
    S : np.ndarray
        Array of support vectors.
    a : np.ndarray
        Coefficients for support vectors.
    g : np.ndarray
        Gradient values for support vectors.
    y : np.ndarray
        Labels of the support vectors.
    b : float
        Bias term.
    delta : float
        Difference in gradient values for convergence checks.
    metrics_tracker : MetricsTracker
        Instance for tracking and visualizing model metrics during training.

    Examples
    --------
    # Basic usage with real-time metrics enabled
    >>> X_train, Y_train = np.array([[1, 2], [2, 3], [3, 4]]), np.array([1, -1, 1])
    >>> X_test, Y_test = np.array([[2, 3], [3, 5]]), np.array([1, -1])
    >>> lasvm = LaSVM(C=1.0, tau=0.01, kernel=RBF(), enable_metrics=True)
    >>> lasvm.fit(X_train, Y_train)
    >>> accuracy = lasvm.score(X_test, Y_test)
    >>> print(f"Test accuracy: {accuracy}")
    """

    def __init__(
        self,
        C,
        tau,
        kernel=RBF(),
        kernelParam=None,
        eps=1e-4,
        verbose=0,
        epoc_offset=2000,
        enable_metrics=True,
    ):
        self.C = C
        self.tau = tau
        self.kernel = kernel
        self.kernelParam = kernelParam
        self.eps = eps
        self.verbose = verbose
        self.epoc_offset = epoc_offset
        self.enable_metrics = enable_metrics

        self.S = np.empty((0, 0))  # Support vectors
        self.a = np.array([])  # Coefficients
        self.g = np.array([])  # Gradient values
        self.y = np.array([])  # Labels
        self.b = 0
        self.delta = 0

        # Initialize the metrics tracker if enabled
        self.metrics_tracker = MetricsTracker() if self.enable_metrics else None

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
        self.S = np.vstack([self.S, v]) if self.S.size else np.array([v])
        self.a = np.append(self.a, 0)
        self.y = np.append(self.y, cls)
        self.g = np.append(self.g, cls - compute_gradient(self.S, self.a, self.b))

        i, j = self._select_extreme_indices(cls)
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
        S = self.S
        kernel_ij = compute_kernel_output(
            self.kernel, S[i].reshape(1, -1), S[j].reshape(1, -1), self.kernelParam
        )
        lambda_val = min(
            (self.g[i] - self.g[j]) / (2 * (1 - kernel_ij)),
            self.C - self.a[i],
            self.a[j],
        )
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
        Y = np.where(Y == 0, -1, Y)
        self.S, self.a, self.g, self.y = (
            np.empty((0, X.shape[1])),
            np.array([]),
            np.array([]),
            np.array([]),
        )

        for i in tqdm(range(X.shape[0])):
            self.lasvm_process(X[i], Y[i])
            if i % self.epoc_offset == 0 or i == X.shape[0] - 1:
                while self.delta > self.tau:
                    self.lasvm_reprocess()
            if self.enable_metrics:
                # Calculate and update metrics periodically during training
                accuracy = self.score(X, Y)
                self.metrics_tracker.update("accuracy", accuracy)
                margin_violation = np.mean(np.abs(self.g) > self.tau)
                self.metrics_tracker.update("margin_violation", margin_violation)

        if self.enable_metrics:
            # Finalize and visualize metrics after training completes
            self.metrics_tracker.finalize()
            self.metrics_tracker.visualize_metrics()

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
        K = compute_kernel_output(self.kernel, self.S, X, self.kernelParam)
        predictions = np.dot(K.T, self.a * self.y) + self.b
        return np.sign(predictions).astype(int)

    def lasvm_reprocess(self):
        """
        Reprocesses support vectors to ensure convergence.
        """
        i, j = self._select_extreme_indices()
        while self.tau_violating(i, j):
            self.lbda(i, j)
            i, j = self._select_extreme_indices()

    def score(self, X, Y):
        """
        Calculate the accuracy of the classifier on the test data.

        Parameters
        ----------
        X : np.ndarray
            Test data, shape (n_samples, n_features).
        Y : np.ndarray
            True labels, shape (n_samples,).

        Returns
        -------
        float
            Accuracy score as the fraction of correct predictions.
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == Y)

        if self.enable_metrics:
            # Track accuracy over time if real-time metrics are enabled
            self.metrics_tracker.update("accuracy", accuracy)

        return accuracy

    def _select_extreme_indices(self, cls=None):
        """
        Helper function to select indices for violating pairs.
        If cls is specified, select based on class label.

        Parameters
        ----------
        cls : int, optional
            Class label for selecting indices.

        Returns
        -------
        tuple of int
            Indices of selected support vectors.
        """
        if cls is not None and cls > 0:
            i = len(self.S) - 1
            _, j = self._find_extreme_indices()
        elif cls is not None:
            j = len(self.S) - 1
            i, _ = self._find_extreme_indices()
        else:
            i, j = self._find_extreme_indices()
        return i, j

    def _find_extreme_indices(self):
        """
        Finds the indices of the support vectors with maximum and minimum gradient values.

        Returns
        -------
        tuple of int
            Indices of support vectors with max and min gradients.
        """
        g_max_idx = np.argmax(self.g)
        g_min_idx = np.argmin(self.g)
        return g_max_idx, g_min_idx

    def tau_violating(self, i, j):
        """
        Checks if the pair (i, j) is violating the tau threshold.

        Parameters
        ----------
        i : int
            Index of the first support vector.
        j : int
            Index of the second support vector.

        Returns
        -------
        bool
            True if pair (i, j) violates tau, False otherwise.
        """
        return abs(self.g[i] - self.g[j]) > self.tau
