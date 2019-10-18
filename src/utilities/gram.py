import numpy as np

class gram():
    def __call__(self, X1,X2,kernel,*args, **kwargs):
        K = np.zeros((len(X1), len(X2)))
        for i in range(len(X1)):
            for j in range(i, len(X1)):
                K[i, j] = kernel(X1[i], X2[j])
                K[j, i] = K[i, j]
        return K

