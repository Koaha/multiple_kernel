import numpy as np

### Operator Classes ###

class OperatorExponential:
    """Exponential Operator: exp(γ * X)"""
    def __call__(self, X, gamma=1.0):
        return np.exp(gamma * X)

class OperatorPolynomial:
    """Polynomial Operator: (X + affine)^degree"""
    def __call__(self, X, affine=0.0, degree=2):
        return (X + affine) ** degree

class OperatorAffine:
    """Affine Combination: a*X + b*Y"""
    def __call__(self, X, Y, a=1.0, b=1.0):
        return a * X + b * Y

class OperatorMultiplication:
    """Element-wise Multiplication: X * Y"""
    def __call__(self, X, Y):
        return X * Y

class OperatorAddition:
    """Element-wise Addition: X + Y"""
    def __call__(self, X, Y):
        return X + Y

class OperatorSubtraction:
    """Element-wise Subtraction: X - Y"""
    def __call__(self, X, Y):
        return X - Y

class OperatorSigmoid:
    """Sigmoid Transformation: 1 / (1 + exp(-γ * X))"""
    def __call__(self, X, gamma=1.0):
        return 1 / (1 + np.exp(-gamma * X))

class OperatorGaussian:
    """Gaussian Transformation: exp(-||X||^2 / (2 * sigma^2))"""
    def __call__(self, X, sigma=1.0):
        norm_squared = np.linalg.norm(X, axis=-1, keepdims=True) ** 2
        return np.exp(-norm_squared / (2 * sigma ** 2))

class OperatorPower:
    """Power Transformation: |X|^d"""
    def __call__(self, X, degree=2.0):
        return np.abs(X) ** degree

class OperatorCosine:
    """Cosine Similarity Operator: cos(γ * X)"""
    def __call__(self, X, gamma=1.0):
        return np.cos(gamma * X)

class OperatorFourierTransform:
    """Fourier Transform Operator: cos(ω * X + φ)"""
    def __call__(self, X, omega=1.0, phi=0.0):
        return np.cos(omega * X + phi)

class OperatorMin:
    """Minimum Operator: element-wise min(X, Y)"""
    def __call__(self, X, Y):
        return np.minimum(X, Y)

class OperatorMax:
    """Maximum Operator: element-wise max(X, Y)"""
    def __call__(self, X, Y):
        return np.maximum(X, Y)

class OperatorDistanceSquared:
    """Distance Squared Operator: ||X - Y||^2"""
    def __call__(self, X, Y):
        return np.sum((X - Y) ** 2, axis=-1, keepdims=True)

class OperatorDistance:
    """Euclidean Distance Operator: ||X - Y||"""
    def __call__(self, X, Y):
        return np.sqrt(np.sum((X - Y) ** 2, axis=-1, keepdims=True))

class OperatorNormalizedDifference:
    """Normalized Difference Operator: (X - Y) / (X + Y + epsilon)"""
    def __call__(self, X, Y, epsilon=1e-8):
        return (X - Y) / (X + Y + epsilon)

class OperatorLogarithmic:
    """Logarithmic Transformation: log(X + offset)"""
    def __call__(self, X, offset=1e-8):
        return np.log(X + offset)

class OperatorReciprocal:
    """Reciprocal Transformation: 1 / (X + epsilon)"""
    def __call__(self, X, epsilon=1e-8):
        return 1 / (X + epsilon)

class OperatorWeightedSum:
    """Weighted Sum: w1*X + w2*Y with weight normalization"""
    def __call__(self, X, Y, w1=0.5, w2=0.5):
        total_weight = w1 + w2
        return (w1 * X + w2 * Y) / total_weight

class OperatorKernelCombination:
    """Combines multiple kernels with weights and returns the weighted sum."""
    def __init__(self, kernels, weights=None):
        self.kernels = kernels
        self.weights = weights or [1 / len(kernels)] * len(kernels)

    def __call__(self, X, Y):
        result = np.zeros((X.shape[0], Y.shape[0]))
        for kernel, weight in zip(self.kernels, self.weights):
            result += weight * kernel(X, Y)
        return result

