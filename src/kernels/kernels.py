import numpy as np
from scipy.spatial.distance import cdist
import scipy.sparse as sp

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
        self.gamma = gamma or 1.0 / (2.0 * sigma ** 2)

    def __call__(self, x, y):
        x, y = np.array(x), np.array(y)
        dists = cdist(x, y, metric="sqeuclidean")
        return np.exp(-self.gamma * dists)

class Laplacian:
    """Laplacian Kernel: K(x, y) = exp(-||x - y|| / sigma)"""
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def __call__(self, x, y):
        dists = cdist(np.array(x), np.array(y), metric="euclidean")
        return np.exp(-dists / self.sigma)

class Chi2Kernel:
    """Chi-square Kernel: K(x, y) = exp(-γ * Σ((x_i - y_i)^2 / (x_i + y_i)))"""
    def __call__(self, X, Y, gamma=1.0):
        X, Y = np.array(X), np.array(Y)
        K = np.zeros((X.shape[0], Y.shape[0]))
        for d in range(X.shape[1]):
            X_col = X[:, d].reshape(-1, 1)
            Y_col = Y[:, d].reshape(-1, 1)
            K += (X_col - Y_col.T) ** 2 / (X_col + Y_col.T)
        return np.exp(-gamma * K)

class MinKernel:
    """Histogram Intersection Kernel: K(x, y) = Σ min(x_i, y_i)"""
    def __call__(self, X, Y):
        K = np.zeros((X.shape[0], Y.shape[0]))
        for d in range(X.shape[1]):
            X_col = X[:, d].reshape(-1, 1)
            Y_col = Y[:, d].reshape(-1, 1)
            K += np.minimum(X_col, Y_col.T)
        return K

class GenMinKernel:
    """Generalized Min Kernel: K(x, y) = Σ min(|x_i|^α, |y_i|^α)"""
    def __call__(self, X, Y, alpha=1.0):
        return MinKernel()(np.abs(X) ** alpha, np.abs(Y) ** alpha)

class Sigmoid:
    """Sigmoid Kernel: K(x, y) = tanh(γ * x · y + coef0)"""
    def __call__(self, x, y, gamma=0.1, coef0=0):
        return np.tanh(gamma * np.dot(x, y.T) + coef0)

class CosineKernel:
    """Cosine Similarity Kernel: K(x, y) = (x · y) / (||x|| * ||y||)"""
    def __call__(self, x, y):
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        y_norm = np.linalg.norm(y, axis=1, keepdims=True)
        return np.dot(x / x_norm, (y / y_norm).T)

class WaveletKernel:
    """Wavelet Kernel: K(x, y) = Σ cos(1.75 * (x_i - y_i) / param) * exp(-((x_i - y_i) / param)^2)"""
    def __call__(self, X, Y, param=1.0):
        X, Y = np.array(X), np.array(Y)
        val = (X[:, np.newaxis, :] - Y[np.newaxis, :, :]) / param
        return np.sum(np.cos(1.75 * val) * np.exp(-val ** 2), axis=2)

class PowerKernel:
    """Power Kernel: K(x, y) = -||x - y||^d"""
    def __call__(self, X, Y, d=1.0):
        dists = cdist(np.array(X), np.array(Y), metric="euclidean")
        return -np.power(dists, d)

class RationalQuadraticKernel:
    """Rational Quadratic Kernel: K(x, y) = 1 - ||x - y||^2 / (||x - y||^2 + α)"""
    def __call__(self, X, Y, alpha=1.0):
        dists = cdist(np.array(X), np.array(Y), metric="sqeuclidean")
        return 1 - dists / (dists + alpha)

class ExponentialChi2Kernel:
    """Exponential Chi-Square Kernel: K(x, y) = exp(-γ * Σ((x_i - y_i)^2 / (x_i + y_i)))"""
    def __call__(self, X, Y, gamma=0.5):
        X, Y = np.array(X), np.array(Y)
        K = np.zeros((X.shape[0], Y.shape[0]))
        for d in range(X.shape[1]):
            X_col = X[:, d].reshape(-1, 1)
            Y_col = Y[:, d].reshape(-1, 1)
            K += (X_col - Y_col.T) ** 2 / (X_col + Y_col.T)
        return np.exp(-gamma * K)

class PeriodicKernel:
    """Periodic Kernel: K(x, y) = exp(-2 * sin^2(π * ||x - y|| / p) / σ^2)"""
    def __init__(self, p=1.0, sigma=1.0):
        self.p = p
        self.sigma = sigma

    def __call__(self, X, Y):
        dists = cdist(X, Y, metric="euclidean")
        return np.exp(-2 * np.sin(np.pi * dists / self.p) ** 2 / (self.sigma ** 2))

class ANOVAKernel:
    """ANOVA Kernel: K(x, y) = Σ (exp(-σ * (x_i - y_i)^2))^d"""
    def __call__(self, X, Y, sigma=0.5, degree=2):
        X, Y = np.array(X), np.array(Y)
        K = np.zeros((X.shape[0], Y.shape[0]))
        for d in range(X.shape[1]):
            X_col = X[:, d].reshape(-1, 1)
            Y_col = Y[:, d].reshape(-1, 1)
            K += (np.exp(-sigma * (X_col - Y_col.T) ** 2)) ** degree
        return K

class FourierKernel:
    """Fourier Kernel: K(x, y) = Σ cos(ω_i · (x - y) + φ_i)"""
    def __call__(self, X, Y, omega=1.0, phi=0.0):
        X, Y = np.array(X), np.array(Y)
        return np.sum(np.cos(omega * (X[:, np.newaxis, :] - Y[np.newaxis, :, :]) + phi), axis=2)

class InverseMultiquadricKernel:
    """Inverse Multiquadric Kernel: K(x, y) = 1 / √(||x - y||^2 + c^2)"""
    def __call__(self, X, Y, c=1.0):
        dists = cdist(np.array(X), np.array(Y), metric="sqeuclidean")
        return 1.0 / np.sqrt(dists + c ** 2)
