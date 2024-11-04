import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm
from src.kernels.kernels import RBF
from src.utilities.optimization_helper import (
    compute_kernel_output,
    compute_margin,
    compute_gradient,
    prune_support_vectors,
    adjust_sets,
    get_min_variation,
)
from src.metrics.metrics_tracker import (
    MetricsTracker,
)  # Import MetricsTracker for real-time metrics


class NySVM(BaseEstimator, ClassifierMixin):
    """
    NySVM is an online Support Vector Machine using the Nyström method for
    efficient training with large datasets. Integrates real-time metric tracking
    to monitor performance during training.

    Parameters
    ----------
    numFeatures : int
        Number of features in the input data.
    C : float, default=1.0
        Regularization parameter.
    eps : float, default=1e-3
        Margin tolerance threshold.
    kernel : callable, default=RBF()
        Kernel function to use. If None, defaults to RBF.
    kernelParam : float, optional
        Parameter for the RBF kernel, if applicable.
    bias : float, default=0.0
        Initial bias term.
    prune : bool, default=False
        If True, enables pruning of support vectors.
    verbose : int, default=0
        Verbosity level. Higher values result in more console output.
    enable_metrics : bool, optional, default=True
        Enables real-time metrics tracking during training if set to True.

    Attributes
    ----------
    numSamplesTrained : int
        Total number of samples trained.
    weights : np.ndarray
        Weight vector associated with support vectors.
    supportSetIndices : list of int
        Indices of support vectors.
    errorSetIndices : list of int
        Indices of error vectors.
    remainderSetIndices : list of int
        Indices of remainder vectors.
    R : np.matrix
        Covariance matrix of support vectors.
    metrics_tracker : MetricsTracker
        Instance for tracking and visualizing model metrics during training.

    Examples
    --------
    # Basic usage with real-time metrics enabled
    >>> X_train, Y_train = np.array([[1, 2], [2, 3], [3, 4]]), np.array([1, -1, 1])
    >>> X_test, Y_test = np.array([[2, 3], [3, 5]]), np.array([1, -1])
    >>> nysvm = NySVM(numFeatures=2, C=1.0, enable_metrics=True)
    >>> nysvm.fit(X_train, Y_train)
    >>> accuracy = nysvm.score(X_test, Y_test)
    >>> print(f"Test accuracy: {accuracy}")
    """

    def __init__(
        self,
        numFeatures,
        C=1,
        eps=1e-3,
        kernel=None,
        kernelParam=None,
        bias=0,
        prune=False,
        verbose=0,
        enable_metrics=True,
    ):
        self.numFeatures = numFeatures
        self.C = C
        self.eps = eps
        self.kernel = kernel or RBF()
        self.kernelParam = kernelParam
        self.bias = bias
        self.prune = prune
        self.verbose = verbose
        self.numSamplesTrained = 0
        self.X, self.Y = [], []
        self.weights = np.array([])
        self.supportSetIndices, self.errorSetIndices, self.remainderSetIndices = (
            [],
            [],
            [],
        )
        self.R = np.matrix([])  # For support vector covariance matrix
        self.enable_metrics = enable_metrics

        # Initialize the metrics tracker if enabled
        self.metrics_tracker = MetricsTracker() if self.enable_metrics else None

    def learn(self, newSampleX, newSampleY):
        """
        Adds a new sample to the model and updates the weight and bias terms
        using the NySVM online learning method.

        Parameters
        ----------
        newSampleX : np.ndarray
            Feature vector of the new sample.
        newSampleY : int
            Label of the new sample.
        """
        self.numSamplesTrained += 1
        self.X.append(newSampleX)
        self.Y.append(newSampleY)
        self.weights = np.append(self.weights, 0)
        i = self.numSamplesTrained - 1

        H = compute_margin(self.weights, np.array(self.X), np.array(self.Y), self.bias)
        if abs(H[i]) <= self.eps:
            self.remainderSetIndices.append(i)
            return

        addNewSample = False
        while not addNewSample:
            beta, gamma = self.computeBetaGamma(i)
            deltaC, flag, minIndex = get_min_variation(H, beta, gamma, i)
            self.update_weights_and_bias(i, deltaC, beta, gamma, H)
            H, addNewSample = adjust_sets(
                H,
                self.weights,
                self.supportSetIndices,
                self.errorSetIndices,
                self.remainderSetIndices,
                flag,
                minIndex,
            )

    def update_weights_and_bias(self, i, deltaC, beta, gamma, H):
        """
        Updates the weights and bias term based on the computed margin.

        Parameters
        ----------
        i : int
            Index of the current sample.
        deltaC : float
            Change in weight.
        beta : np.ndarray
            Beta values for the current sample.
        gamma : np.ndarray
            Gamma values for the current sample.
        H : np.ndarray
            Margin array.
        """
        self.weights[i] += deltaC
        delta = beta * deltaC
        self.bias += delta.item(0)
        weight_delta = np.array(delta[1:]).reshape(-1)
        self.weights[self.supportSetIndices] += weight_delta
        H += gamma * deltaC

    def fit(self, X, Y):
        """
        Fits the NySVM model to the provided data using online training.

        Parameters
        ----------
        X : np.ndarray
            Training data, shape (n_samples, n_features).
        Y : np.ndarray
            Training labels, shape (n_samples,).
        """
        self.numFeatures = X.shape[1]
        X, Y = np.array(X), np.array(Y)
        Y[Y == 0] = -1  # Convert 0 labels to -1 for SVM
        for i in tqdm(range(X.shape[0])):
            self.learn(X[i, :], Y[i])
            if self.prune:
                self.prune_vector()

            if self.enable_metrics:
                accuracy = self.score(X, Y)
                margin_violation = np.mean(
                    np.abs(
                        compute_margin(
                            self.weights, np.array(self.X), np.array(self.Y), self.bias
                        )
                    )
                    > self.eps
                )
                self.metrics_tracker.update("accuracy", accuracy)
                self.metrics_tracker.update("margin_violation", margin_violation)

        if self.enable_metrics:
            # Finalize and visualize metrics after training completes
            self.metrics_tracker.finalize()
            self.metrics_tracker.visualize_metrics()

    def prune_vector(self, threshold=1e-4):
        """
        Prunes less significant support vectors based on weight magnitude.

        Parameters
        ----------
        threshold : float, default=1e-4
            Threshold for pruning. Support vectors with weights below this
            value will be pruned.
        """
        self.supportSetIndices = prune_support_vectors(
            self.weights, self.supportSetIndices, threshold
        )

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
        K = compute_kernel_output(self.kernel, np.array(self.X), X, self.kernelParam)
        predictions = K.T.dot(self.weights.reshape(-1, 1)) + self.bias
        return np.where(predictions > 0, 1, -1).astype(int)

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

    def computeBetaGamma(self, i):
        """
        Computes beta and gamma values for margin adjustments for a sample.

        Parameters
        ----------
        i : int
            Index of the sample to compute beta and gamma.

        Returns
        -------
        beta : np.ndarray
            Computed beta values.
        gamma : np.ndarray
            Computed gamma values.
        """
        Qsi = compute_kernel_output(
            self.kernel,
            np.array(self.X)[self.supportSetIndices],
            np.array(self.X)[i],
            self.kernelParam,
        )
        beta = (
            -self.R @ np.append(np.matrix([1]), Qsi, axis=0)
            if len(self.supportSetIndices) > 0
            else np.array([])
        )
        Qxi = compute_kernel_output(
            self.kernel, np.array(self.X), np.array(self.X)[i], self.kernelParam
        )
        Qxs = compute_kernel_output(
            self.kernel,
            np.array(self.X),
            np.array(self.X)[self.supportSetIndices],
            self.kernelParam,
        )
        gamma = (
            Qxi + np.append(np.ones([self.numSamplesTrained, 1]), Qxs, 1) @ beta
            if len(self.supportSetIndices) > 0
            else np.array(np.ones_like(Qxi))
        )
        return np.nan_to_num(beta), np.nan_to_num(gamma)
