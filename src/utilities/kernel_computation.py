import numpy as np
from src.kernels.kernels import RBF


class KernelComputation:
    """
    A class for computing kernel-based operations.

    Parameters
    ----------
    kernel : callable
        The kernel function to use, e.g., RBF, Linear, Polynomial.

    kernel_param : float, optional (default=None)
        Additional parameter for the kernel, such as gamma for RBF kernels.

    Methods
    -------
    compute_output(X, Y)
        Computes the kernel output matrix between X and Y.
    """

    def __init__(self, kernel, kernel_param=None):
        self.kernel = kernel
        self.kernel_param = kernel_param

    def compute_output(self, X, Y):
        """
        Computes the kernel output matrix between X and Y.

        Parameters
        ----------
        X : np.ndarray
            Input matrix with shape (n_samples_X, n_features).

        Y : np.ndarray
            Input matrix with shape (n_samples_Y, n_features).

        Returns
        -------
        np.ndarray
            The kernel output matrix of shape (n_samples_X, n_samples_Y).
        """
        if isinstance(self.kernel, RBF):
            return self.kernel(X, Y, gamma=self.kernel_param)
        return self.kernel(X, Y)
