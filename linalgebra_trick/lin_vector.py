import numpy as np

def linear_combination_of_linear_combination():
    dim = 2
    m = 3
    k = 4
    a = np.random.rand(dim,m)
    A = np.random.rand(m,k) #matrix of coef alpha
    B = a@A
    beta = np.random.rand(k,1)
    c = B@beta
    w = beta.T@A.T
    c_alter = a@w.T

if __name__ == "__main__":
    linear_combination_of_linear_combination()