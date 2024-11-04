import numpy as np
from scipy.spatial.distance import cdist

### Kernel Classes ###


class Linear:
    """Linear Kernel: K(x, y) = x · y"""

    def __call__(self, a, b):
        return np.dot(np.array(a), np.array(b).T)


class Polynomial:
    """Polynomial Kernel: K(x, y) = (γ * (x · y) + coef0)^degree"""

    def __call__(self, a, b, degree=3, gamma=None, coef0=1):
        gamma = gamma or (1 / len(a))
        return (coef0 + gamma * np.dot(np.array(a), np.array(b).T)) ** degree


class RBF:
    """Radial Basis Function (Gaussian) Kernel: K(x, y) = exp(-γ ||x - y||^2)"""

    def __init__(self, sigma=1.0, gamma=None):
        self.gamma = gamma or 1.0 / (2.0 * sigma**2)

    def __call__(self, x, y):
        dists = cdist(np.array(x), np.array(y), metric="sqeuclidean")
        return np.exp(-self.gamma * dists)


class Sigmoid:
    """Sigmoid Kernel: K(x, y) = tanh(γ * x · y + coef0)"""

    def __call__(self, x, y, gamma=0.1, coef0=0):
        return np.tanh(gamma * np.dot(np.array(x), np.array(y).T) + coef0)


class WaveletKernel:
    """Wavelet Kernel: K(x, y) = Σ cos(1.75 * (x_i - y_i) / param) * exp(-((x_i - y_i) / param)^2)"""

    def __call__(self, X, Y, param=1.0):
        val = (X[:, np.newaxis, :] - Y[np.newaxis, :, :]) / param
        return np.sum(np.cos(1.75 * val) * np.exp(-(val**2)), axis=2)


class StringKernel:
    """String Kernel: Counts common substrings between sequences."""

    def __call__(self, X, Y, substring_length=3):
        def count_common_substrings(s1, s2):
            return sum(
                s1[i : i + substring_length] in s2
                for i in range(len(s1) - substring_length + 1)
            )

        K = np.zeros((len(X), len(Y)))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                K[i, j] = count_common_substrings(x, y)
        return K


class CustomKernel:
    """Allows for custom kernel functions."""

    def __init__(self, custom_func):
        self.custom_func = custom_func

    def __call__(self, X, Y, **kwargs):
        K = np.array([[self.custom_func(x, y, **kwargs) for y in Y] for x in X])
        return K


class Laplacian:
    """Laplacian Kernel: K(x, y) = exp(-||x - y|| / sigma)"""

    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def __call__(self, x, y):
        dists = cdist(np.array(x), np.array(y), metric="euclidean")
        return np.exp(-dists / self.sigma)


class Chi2Kernel:
    """Chi-square Kernel with efficient vectorization."""

    def __call__(self, X, Y, gamma=1.0):
        X, Y = np.array(X), np.array(Y)
        K = np.zeros((X.shape[0], Y.shape[0]))
        for d in range(X.shape[1]):
            X_col = X[:, d][:, np.newaxis]
            Y_col = Y[:, d][np.newaxis, :]
            K += (X_col - Y_col) ** 2 / (X_col + Y_col + 1e-10)
        return np.exp(-gamma * K)


class MinKernel:
    """Histogram Intersection Kernel: Uses element-wise min."""

    def __call__(self, X, Y):
        X, Y = np.array(X), np.array(Y)
        return np.sum(np.minimum(X[:, np.newaxis, :], Y[np.newaxis, :, :]), axis=2)


class GenMinKernel:
    """Generalized Min Kernel using element-wise min with a power."""

    def __call__(self, X, Y, alpha=1.0):
        return MinKernel()(np.abs(X) ** alpha, np.abs(Y) ** alpha)


class CosineKernel:
    """Cosine Similarity Kernel: K(x, y) = (x · y) / (||x|| * ||y||)"""

    def __call__(self, x, y):
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        y_norm = np.linalg.norm(y, axis=1, keepdims=True)
        return np.dot(x / x_norm, (y / y_norm).T)


class PowerKernel:
    """Power Kernel with element-wise power."""

    def __call__(self, X, Y, d=1.0):
        dists = cdist(np.array(X), np.array(Y), metric="euclidean")
        return -np.power(dists, d)


class RationalQuadraticKernel:
    """Rational Quadratic Kernel."""

    def __call__(self, X, Y, alpha=1.0):
        dists = cdist(np.array(X), np.array(Y), metric="sqeuclidean")
        return 1 - dists / (dists + alpha)


class PeriodicKernel:
    """Periodic Kernel with element-wise sin."""

    def __init__(self, p=1.0, sigma=1.0):
        self.p = p
        self.sigma = sigma

    def __call__(self, X, Y):
        dists = cdist(X, Y, metric="euclidean")
        return np.exp(-2 * np.sin(np.pi * dists / self.p) ** 2 / (self.sigma**2))


class FourierKernel:
    """Fourier Kernel using element-wise cos."""

    def __call__(self, X, Y, omega=1.0, phi=0.0):
        return np.sum(
            np.cos(omega * (X[:, np.newaxis, :] - Y[np.newaxis, :, :]) + phi), axis=2
        )


class InverseMultiquadricKernel:
    """Inverse Multiquadric Kernel with efficient computation."""

    def __call__(self, X, Y, c=1.0):
        dists = cdist(np.array(X), np.array(Y), metric="sqeuclidean")
        return 1.0 / np.sqrt(dists + c**2)
