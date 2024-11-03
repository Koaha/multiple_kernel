import numpy as np

class Pegasos:
    """
    Pegasos Solver for SVM using stochastic sub-gradient descent.
    
    Supports both linear and kernelized SVM, allowing efficient optimization
    with large-scale data through stochastic updates.

    Parameters
    ----------
    X : np.ndarray
        Training data, shape (n_samples, n_features).
    Y : np.ndarray
        Training labels, shape (n_samples,).
    lambda_param : float, optional
        Regularization parameter. Default is 1.0.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    kernel : callable, optional
        Kernel function for non-linear SVM. If None, a linear SVM is used.
    
    Attributes
    ----------
    w : np.ndarray
        Weight vector for linear SVM.
    alpha : np.ndarray
        Dual coefficients for support vectors (only for kernelized SVM).
    """

    def __init__(self, X, Y, lambda_param=1.0, max_iter=1000, kernel=None):
        self.X = X
        self.Y = self._convert_labels(Y)
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.kernel = kernel if kernel else self.linear_kernel

        # Initialize model parameters
        if kernel:
            self.alpha = np.zeros(len(Y))  # Alpha coefficients for kernelized SVM
        else:
            self.w = np.zeros(X.shape[1])  # Weight vector for linear SVM

    def linear_kernel(self, x1, x2):
        """
        Linear kernel function.

        Parameters
        ----------
        x1 : np.ndarray
            First input vector.
        x2 : np.ndarray
            Second input vector.

        Returns
        -------
        float
            Dot product of x1 and x2.
        """
        return np.dot(x1, x2)

    def _convert_labels(self, Y):
        """
        Converts labels to {-1, 1} format required for Pegasos.

        Parameters
        ----------
        Y : np.ndarray
            Original labels.

        Returns
        -------
        np.ndarray
            Converted labels.
        """
        return np.where(Y == 0, -1, Y)

    def fit(self):
        """
        Train the SVM model using the Pegasos algorithm.
        
        For linear SVM, updates the weight vector w. For kernelized SVM,
        updates dual coefficients alpha using the kernel trick.
        """
        for t in range(1, self.max_iter + 1):
            # Select a random sample
            i = np.random.randint(len(self.Y))
            eta_t = 1 / (self.lambda_param * t)

            if self.kernel:
                # Kernelized SVM update
                self._update_alpha(i, eta_t)
            else:
                # Linear SVM update
                self._update_w(i, eta_t)

    def _update_w(self, i, eta_t):
        """
        Update the weight vector for linear SVM.

        Parameters
        ----------
        i : int
            Index of the randomly selected sample.
        eta_t : float
            Step size for the current iteration.
        """
        x_i, y_i = self.X[i], self.Y[i]
        margin = y_i * np.dot(self.w, x_i)
        if margin < 1:
            self.w = (1 - eta_t * self.lambda_param) * self.w + eta_t * y_i * x_i
        else:
            self.w = (1 - eta_t * self.lambda_param) * self.w

    def _update_alpha(self, i, eta_t):
        """
        Update the alpha coefficients for kernelized SVM.

        Parameters
        ----------
        i : int
            Index of the randomly selected sample.
        eta_t : float
            Step size for the current iteration.
        """
        x_i, y_i = self.X[i], self.Y[i]
        margin = y_i * self._predict_kernel(x_i)
        if margin < 1:
            self.alpha[i] += 1  # Increment alpha only if margin constraint is violated

    def _predict_kernel(self, x):
        """
        Calculate the decision function for kernelized SVM.

        Parameters
        ----------
        x : np.ndarray
            Input vector.

        Returns
        -------
        float
            Decision function value.
        """
        K = np.array([self.kernel(x_i, x) for x_i in self.X])
        return np.dot(self.alpha * self.Y, K)

    def predict(self, X):
        """
        Predict the class labels for samples in X.

        Parameters
        ----------
        X : np.ndarray
            Data to be classified, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted class labels, shape (n_samples,).
        """
        if self.kernel:
            return np.sign([self._predict_kernel(x) for x in X])
        else:
            return np.sign(np.dot(X, self.w))

    def score(self, X, Y):
        """
        Calculate the accuracy of the classifier.

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
        return np.mean(predictions == self._convert_labels(Y))
