import numpy as np
from sklearn.cluster import KMeans


class NystromApproximation:
    """
    Computes the Nyström approximation of a kernel matrix, enabling efficient
    kernel computation for large datasets.

    Parameters
    ----------
    X : np.ndarray
        Input data, shape (n_samples, n_features).
    kernel : callable
        Kernel function to use. This function should accept two samples and
        return a scalar similarity measure.
    sample_size : int
        Number of samples to use for approximation, where sample_size < n_samples.
    sampling_method : str, default='random'
        Sampling method to use for selecting the subset. Options are:
        - 'random': Random sampling.
        - 'kmeans': K-means clustering to select centroids as subset.
        - 'support_vectors': Use a predefined set of indices for support vectors.
    regularization : float, default=1e-5
        Regularization parameter added to the diagonal of K_mm to prevent
        issues with near-singular matrices.
    support_indices : np.ndarray, optional
        Predefined indices for support vectors (only used if sampling_method='support_vectors').
    """

    def __init__(
        self,
        X,
        kernel,
        sample_size,
        sampling_method="random",
        regularization=1e-5,
        support_indices=None,
    ):
        self.X = X
        self.kernel = kernel
        self.sample_size = sample_size
        self.sampling_method = sampling_method
        self.regularization = regularization
        self.support_indices = support_indices
        self.selected_indices = self._select_samples()

    def _select_samples(self):
        """
        Selects a subset of samples based on the specified sampling method.

        Returns
        -------
        np.ndarray
            Indices of selected samples.
        """
        n_samples = len(self.X)

        if self.sampling_method == "random":
            return np.random.choice(n_samples, self.sample_size, replace=False)

        elif self.sampling_method == "kmeans":
            kmeans = KMeans(n_clusters=self.sample_size, random_state=0).fit(self.X)
            centers = kmeans.cluster_centers_
            # Find closest points to each center for representative sampling
            selected_indices = np.argmin(cdist(self.X, centers), axis=0)
            return np.unique(selected_indices)[: self.sample_size]

        elif (
            self.sampling_method == "support_vectors"
            and self.support_indices is not None
        ):
            if len(self.support_indices) < self.sample_size:
                raise ValueError("Not enough support indices provided.")
            return np.array(self.support_indices[: self.sample_size])

        else:
            raise ValueError(
                "Invalid sampling method or missing required support_indices."
            )

    def compute_nystrom_approximation(self):
        """
        Computes the Nyström approximation for the kernel matrix of X.

        Returns
        -------
        K_approx : np.ndarray
            Approximate kernel matrix, shape (n_samples, n_samples).
        """
        X_m = self.X[self.selected_indices]

        # Compute sub-kernel matrix K_mm for selected samples with regularization
        K_mm = self._compute_kernel_matrix(X_m, X_m)
        K_mm += self.regularization * np.eye(self.sample_size)

        # Compute cross-kernel matrix K_nm between all samples and selected samples
        K_nm = self._compute_kernel_matrix(self.X, X_m)

        # Invert K_mm (regularization already added)
        K_mm_inv = np.linalg.pinv(K_mm)

        # Compute the Nyström approximation
        K_approx = K_nm @ K_mm_inv @ K_nm.T
        return K_approx

    def _compute_kernel_matrix(self, X1, X2):
        """
        Computes the kernel matrix between two sets of samples using the specified kernel.

        Parameters
        ----------
        X1 : np.ndarray
            First set of samples.
        X2 : np.ndarray
            Second set of samples.

        Returns
        -------
        K : np.ndarray
            Kernel matrix of shape (n_samples_1, n_samples_2).
        """
        # Use broadcasting and vectorization to avoid explicit loops
        return self.kernel(X1, X2)
