import numpy as np
import cvxpy as cp


class CP:
    """
    Cutting Plane (CP) solver for SVM optimization, supporting both primal and dual formulations.

    Parameters
    ----------
    mode : str, optional
        Specify the solver mode, either 'primal' or 'dual'. Default is 'primal'.

    Attributes
    ----------
    w : np.ndarray
        Weight vector after training (primal).
    b : float
        Bias term after training (primal).
    alpha : np.ndarray
        Dual coefficients for support vectors (dual).
    C_matrix : np.ndarray
        Subset of columns from kernel matrix (dual, Nyström).
    W_matrix : np.ndarray
        Positive semidefinite matrix for kernel approximation (dual, Nyström).
    """

    def __init__(self, mode="primal"):
        if mode not in ["primal", "dual"]:
            raise ValueError("Mode must be 'primal' or 'dual'")
        self.mode = mode
        self.w = None
        self.b = None
        self.alpha = None
        self.C_matrix = None
        self.W_matrix = None

    def fit(
        self, X, Y, C=1.0, epsilon=1e-4, s=10, kernel_func=None, accelerated_flag=0
    ):
        """
        Fit the model using the specified solver mode (primal or dual).

        Parameters
        ----------
        X : np.ndarray
            Training data, shape (n_samples, n_features).
        Y : np.ndarray
            Training labels, shape (n_samples,).
        C : float
            Regularization parameter.
        epsilon : float
            Tolerance level for optimization.
        s : int
            Number of samples for Nyström approximation (used only in dual).
        kernel_func : callable, optional
            Kernel function for the dual formulation. Required if mode is 'dual'.
        accelerated_flag : int
            Flag for enabling acceleration (used only in dual).
        """
        if self.mode == "primal":
            self._fit_primal(X, Y, C, epsilon)
        elif self.mode == "dual":
            if kernel_func is None:
                raise ValueError("A kernel function must be provided in dual mode.")
            self._fit_dual(X, Y, s, kernel_func, C, accelerated_flag)

    def _fit_primal(self, X, Y, C, epsilon):
        """Primal cutting plane SVM solver using cvxpy with vectorized constraints."""
        n_samples, n_features = X.shape
        w = cp.Variable(n_features)
        si = cp.Variable(n_samples)
        obj = cp.Minimize(0.5 * cp.norm(w) ** 2 + C * cp.sum(si))
        constraints = [Y * (X @ w) >= 1 - si, si >= 0]

        problem = cp.Problem(obj, constraints)
        problem.solve()

        # Retrieve weight and bias
        self.w = w.value
        self.b = self._compute_bias(X, Y)

    def _fit_dual(self, X, Y, s, kernel_func, C, accelerated_flag):
        """Dual cutting plane solver using Nyström approximation with vectorized kernel calculations."""
        n_samples = len(Y)
        indices = np.random.choice(n_samples, s, replace=False)
        X_subset = X[indices]

        # Vectorized computation of C_matrix and W_matrix
        C_matrix = kernel_func(X_subset, X)
        W_matrix = C_matrix[:, indices]

        alpha = cp.Variable(n_samples)
        kernel_matrix = C_matrix @ W_matrix.T
        obj = cp.Maximize(
            cp.sum(alpha) - 0.5 * cp.quad_form(cp.multiply(Y, alpha), kernel_matrix)
        )
        constraints = [alpha >= 0, alpha <= C, cp.sum(cp.multiply(Y, alpha)) == 0]

        # Solve the dual problem
        problem = cp.Problem(obj, constraints)
        problem.solve()

        # Store results for prediction
        self.alpha = alpha.value
        self.C_matrix = C_matrix
        self.W_matrix = W_matrix

    def _compute_bias(self, X, Y):
        """Compute bias based on support vectors in primal mode."""
        decision_values = (self.w @ X.T) * Y
        support_vectors_idx = np.where(np.isclose(decision_values, 1, atol=1e-5))[0]
        if support_vectors_idx.size > 0:
            b = np.mean(Y[support_vectors_idx] - X[support_vectors_idx] @ self.w)
        else:
            b = 0.0
        return b

    def predict(self, X, kernel_func=None):
        """
        Predict labels for samples in X.

        Parameters
        ----------
        X : np.ndarray
            Data to be classified, shape (n_samples, n_features).
        kernel_func : callable, optional
            Kernel function for prediction (required for dual mode).

        Returns
        -------
        np.ndarray
            Predicted class labels, shape (n_samples,).
        """
        if self.mode == "primal":
            if self.w is None or self.b is None:
                raise ValueError("Model has not been trained in primal mode.")
            return np.sign(X @ self.w + self.b).astype(int)
        elif self.mode == "dual":
            if self.alpha is None or self.C_matrix is None or self.W_matrix is None:
                raise ValueError("Model has not been trained in dual mode.")
            if kernel_func is None:
                raise ValueError("A kernel function must be provided in dual mode.")
            return self._predict_dual(X, kernel_func)

    def _predict_dual(self, X, kernel_func):
        """Predict using dual mode with Nyström approximation."""
        # Vectorized computation of the approximate kernel matrix K
        K = kernel_func(X, self.C_matrix)
        decision_values = (self.alpha * self.C_matrix) @ K
        return np.sign(decision_values).astype(int)
