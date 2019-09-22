import numpy as np

class operator_exponential():
    def __call__(self, X, gamma=1):
        if gamma > 0:
            return np.exp(gamma * X)
        return np.exp(X)

class operator_polynomial():
    def __call__(self, X, affine=0, degree=4):
        if (affine >= 0) & (degree > 0):
            return (X+affine)**degree
        return (X) ** 1

class operator_Fourier():
    def __call__(self, X):
        # TODO
        return X

class operator_affine():
    def __call__(self, X, Y, a=1, b=1):
        if (a>0) & (b>0):
            return a*X+b*Y
        return X + Y

class operator_multiplication():
    def __call__(self, X, Y):
        return X*Y