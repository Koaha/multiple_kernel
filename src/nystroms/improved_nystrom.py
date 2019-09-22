import numpy as np
import scipy.spatial.distance as dst
import scipy.linalg as la
from scipy import sparse
import os,sys
sys.path.append("..")
from kernels.kernels import RBF,Linear

class nystorm():

    """
       % INys.m implments the improved nystrom low-rank approximation method in
    % <Improved Nystrom low-rank Approximation and Error Analysis> by Zhang and
    % Kwok, ICML 2008

    %Input:
    :param data: n-by-dim data matrix;
    :param m: number of landmark points;
    :param kernel: (struct) kernel type and parameter
    :param s: 'r' for random sampling and 'k' for k-means based sampling

    %Output:
    :return Ktilde: approximation of kernel matrix K, in the form of GG'

    """
    def INyStrom(self, kernel, data, m, s):
        n,dim = data.shape
        if (s == 'k'):
            idx, center, m = self.eff_kmeans(data, m, 5)
        if (s == 'r'):
            dex = np.random.permutation(n)
            center = data[dex[1:m],:]
        W = kernel(center,center)
        E = kernel(center,data)

        [Va, Ve] = np.linalg.eig(W);
        # va = np.diag(Va);
        va = Va
        pidx = np.where(va > 1e-6);
        # inVa = np.sparse(np.diag(np.power(va(pidx),(-0.5))));
        inVa = sparse.csc_matrix(np.diag(np.power(va[pidx],(-0.5))))
        # inVa = (np.diag(np.power(va[pidx], (-0.5))))
        # G = E @ Ve[:, pidx] @ inVa;
        G = E.T @ Ve[:, pidx[0]] @ inVa
        return G @ G.T

    def eff_kmeans(self, data, m, max_iter):
        n,dim = data.shape
        dex = np.random.permutation(n)
        center = data[dex[:m],:]
        for i in range(max_iter):
            nul = np.zeros(m)
            xx, idx = min(dst.cdist(center,data))
            for j in range(m):
                dex = np.where(idx == j)
                l = len(dex)
                cltr = data[dex,:]
                if l >1:
                    center[j,:]  = np.mean(cltr)
                elif l == 1:
                    center[j,:] = cltr
                else:
                    nul[j] = 1
        dex = np.where(nul == 0)
        m = len(dex)
        center = center[dex,:]
        return center

if __name__ == "__main__":
    A = np.random.rand(10000,10)
    ny = nystorm()
    rbf = RBF()
    linear = Linear()

    nystrom_output = ny.INyStrom(rbf,A,1000,'r')
    kernel_output = rbf(A,A)

    ker_diff = np.abs(nystrom_output-kernel_output)
    print(ker_diff)
    print(sum(ker_diff))
