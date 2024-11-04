import numpy as np


class Gram:
    """
    Gram computes the kernel matrix between two sets of samples, X1 and X2,
    using a specified kernel function. The kernel can be any callable function
    that takes two samples and returns a scalar value.

    Parameters
    ----------
    kernel : callable
        Kernel function to compute the similarity between two samples.
    *args, **kwargs :
        Additional arguments passed to the kernel function.

    Methods
    -------
    __call__(X1, X2, kernel, *args, **kwargs)
        Computes the Gram (kernel) matrix for X1 and X2 using the specified kernel.
    """

    def __call__(self, X1, X2, kernel, *args, **kwargs):
        """
        Computes the kernel matrix (Gram matrix) between two sets of samples.

        Parameters
        ----------
        X1 : np.ndarray
            First set of samples, shape (n_samples_1, n_features).
        X2 : np.ndarray
            Second set of samples, shape (n_samples_2, n_features).
        kernel : callable
            Kernel function to use. The function should accept two samples and
            return a scalar similarity measure.
        *args, **kwargs :
            Additional arguments passed to the kernel function.

        Returns
        -------
        np.ndarray
            Kernel matrix of shape (n_samples_1, n_samples_2).
        """
        K = np.zeros((len(X1), len(X2)))
        for i in range(len(X1)):
            for j in range(i, len(X2)):
                K[i, j] = kernel(X1[i], X2[j], *args, **kwargs)
                K[j, i] = K[i, j]  # Symmetric assignment for efficiency
        return K
