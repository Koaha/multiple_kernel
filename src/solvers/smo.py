import numpy as np


class SMO:
    """
    Sequential Minimal Optimization (SMO) solver for Support Vector Machine (SVM).

    This solver uses the SMO algorithm to solve the dual optimization problem for SVMs.
    It supports both linear and non-linear kernels, error caching, and efficient alpha-pair selection
    for faster convergence.

    Parameters
    ----------
    X : np.ndarray
        Training data, shape (n_samples, n_features).
    Y : np.ndarray
        Training labels, shape (n_samples,).
    C : float, optional
        Regularization parameter. Default is 1.0.
    tol : float, optional
        Tolerance for stopping criterion. Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    kernel : callable, optional
        Kernel function for non-linear SVM. Default is linear kernel.

    Attributes
    ----------
    alpha : np.ndarray
        Dual coefficients for support vectors.
    b : float
        Bias term.
    errors : np.ndarray
        Error cache to store error terms for each sample.
    """

    def __init__(self, X, Y, C=1.0, tol=1e-4, max_iter=1000, kernel=None):
        self.X = X
        self.Y = Y
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.kernel = kernel if kernel else self.linear_kernel
        self.alpha = np.zeros(len(Y))
        self.b = 0
        self.errors = np.zeros(
            len(Y)
        )  # Error cache to store error terms for each sample

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

    def compute_kernel_matrix(self):
        """
        Compute the kernel matrix for the training data.

        Returns
        -------
        np.ndarray
            Kernel matrix, shape (n_samples, n_samples).
        """
        return np.array([[self.kernel(x1, x2) for x2 in self.X] for x1 in self.X])

    def fit(self):
        """
        Train the SVM model using the SMO algorithm.

        Notes
        -----
        The method iteratively updates the dual variables (alphas) using the SMO algorithm.
        It stops when no significant changes are observed in the alpha values or when the
        maximum number of iterations is reached.
        """
        kernel_matrix = self.compute_kernel_matrix()
        examine_all, num_changed, iter_count = True, 0, 0

        while (num_changed > 0 or examine_all) and iter_count < self.max_iter:
            num_changed = 0
            indices = (
                range(len(self.Y))
                if examine_all
                else np.where((self.alpha != 0) & (self.alpha != self.C))[0]
            )
            num_changed = sum(self._examine_example(i, kernel_matrix) for i in indices)
            iter_count += 1
            examine_all = not examine_all if num_changed == 0 else examine_all

    def _examine_example(self, i, kernel_matrix):
        """
        Examines and updates a single example to modify alphas.

        Parameters
        ----------
        i : int
            Index of the first alpha (alpha_i).
        kernel_matrix : np.ndarray
            Precomputed kernel matrix for training data.

        Returns
        -------
        int
            1 if alpha values were updated, 0 otherwise.
        """
        E_i = self._error(i, kernel_matrix)
        r_i = E_i * self.Y[i]

        if (r_i < -self.tol and self.alpha[i] < self.C) or (
            r_i > self.tol and self.alpha[i] > 0
        ):
            j = self._select_second_alpha(i, E_i)
            return self._update_alphas(i, j, kernel_matrix)
        return 0

    def _update_alphas(self, i, j, kernel_matrix):
        """
        Perform SMO update on alpha[i] and alpha[j].

        Parameters
        ----------
        i : int
            Index of the first alpha (alpha_i).
        j : int
            Index of the second alpha (alpha_j).
        kernel_matrix : np.ndarray
            Precomputed kernel matrix for training data.

        Returns
        -------
        bool
            True if alphas were updated successfully, False otherwise.
        """
        if i == j:
            return False

        alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]
        E_i, E_j = self._error(i, kernel_matrix), self._error(j, kernel_matrix)
        L, H = self._compute_L_H(i, j)

        if L == H:
            return False

        eta = 2.0 * kernel_matrix[i, j] - kernel_matrix[i, i] - kernel_matrix[j, j]
        if eta >= 0:
            return False

        # Update alpha_j and clip to bounds L and H
        self.alpha[j] -= (self.Y[j] * (E_i - E_j)) / eta
        self.alpha[j] = np.clip(self.alpha[j], L, H)

        # Check if change in alpha_j is too small
        if abs(self.alpha[j] - alpha_j_old) < 1e-5:
            return False

        # Update alpha_i
        self.alpha[i] += self.Y[i] * self.Y[j] * (alpha_j_old - self.alpha[j])

        # Update bias term
        self._update_bias(E_i, E_j, i, j, alpha_i_old, alpha_j_old, kernel_matrix)

        # Update error cache
        self.errors[i] = self._error(i, kernel_matrix)
        self.errors[j] = self._error(j, kernel_matrix)

        return True

    def _compute_L_H(self, i, j):
        """
        Compute bounds L and H for alpha[j].

        Parameters
        ----------
        i : int
            Index of the first alpha.
        j : int
            Index of the second alpha.

        Returns
        -------
        tuple
            Lower and upper bounds (L, H).
        """
        if self.Y[i] != self.Y[j]:
            return max(0, self.alpha[j] - self.alpha[i]), min(
                self.C, self.C + self.alpha[j] - self.alpha[i]
            )
        else:
            return max(0, self.alpha[i] + self.alpha[j] - self.C), min(
                self.C, self.alpha[i] + self.alpha[j]
            )

    def _error(self, i, kernel_matrix):
        """
        Compute error for the ith example.

        Parameters
        ----------
        i : int
            Index of the example.
        kernel_matrix : np.ndarray
            Precomputed kernel matrix for training data.

        Returns
        -------
        float
            Error term for the ith example.
        """
        return np.dot((self.alpha * self.Y), kernel_matrix[:, i]) + self.b - self.Y[i]

    def _update_bias(self, E_i, E_j, i, j, alpha_i_old, alpha_j_old, kernel_matrix):
        """
        Update the bias term after updating alphas.

        Parameters
        ----------
        E_i : float
            Error for the first example.
        E_j : float
            Error for the second example.
        i : int
            Index of the first example.
        j : int
            Index of the second example.
        alpha_i_old : float
            Previous value of alpha[i].
        alpha_j_old : float
            Previous value of alpha[j].
        kernel_matrix : np.ndarray
            Precomputed kernel matrix for training data.
        """
        b1 = (
            self.b
            - E_i
            - self.Y[i] * (self.alpha[i] - alpha_i_old) * kernel_matrix[i, i]
            - self.Y[j] * (self.alpha[j] - alpha_j_old) * kernel_matrix[i, j]
        )
        b2 = (
            self.b
            - E_j
            - self.Y[i] * (self.alpha[i] - alpha_i_old) * kernel_matrix[i, j]
            - self.Y[j] * (self.alpha[j] - alpha_j_old) * kernel_matrix[j, j]
        )

        if 0 < self.alpha[i] < self.C:
            self.b = b1
        elif 0 < self.alpha[j] < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2

    def _select_second_alpha(self, i, E_i):
        """
        Select second alpha (j) using max-error heuristic.

        Parameters
        ----------
        i : int
            Index of the first alpha (alpha_i).
        E_i : float
            Error for the first example.

        Returns
        -------
        int
            Index of the second alpha (alpha_j).
        """
        non_bound_indices = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
        if len(non_bound_indices) > 1:
            j = non_bound_indices[
                np.argmax(np.abs(E_i - self.errors[non_bound_indices]))
            ]
        else:
            j = i
            while j == i:
                j = np.random.randint(len(self.Y))
        return j

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
        K = np.array([[self.kernel(x_i, x_j) for x_j in self.X] for x_i in X])
        decision = np.dot((self.alpha * self.Y), K.T) + self.b
        return np.sign(decision)
