import numpy as np
import scipy.spatial.distance as dst
import scipy.linalg as la
from scipy import sparse
import os,sys
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae

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
    A = np.random.rand(15000,10)
    B = np.random.rand(15000,10)
    ny = nystorm()
    rbf = RBF()
    linear = Linear()

    samples_size_set = np.arange(100,1000,300)
    run_time_nyS = []
    run_time_kerS = []
    acc_list = []

    for  sample_size in samples_size_set:
        print("--- compute the nystrom approximation ---")
        start_time_ny = time.time()
        # nystrom_output = ny.INyStrom(rbf,A,100,'r')
        nystrom_output = ny.INyStrom(rbf, A, sample_size, 'r')
        run_time_ny = (time.time() - start_time_ny)
        print("--- %s seconds ---" % run_time_ny)
        # nystrom_output = []

        print("--- compute the origin approximation ---")
        start_time_ker = time.time()
        kernel_output = rbf(A,A)
        run_time_ker = (time.time() - start_time_ker)
        print("--- %s seconds ---" % run_time_ker)
        mae_res = mae(kernel_output,nystrom_output)
        acc_list.append(mae_res)
        run_time_kerS.append(run_time_ker)
        run_time_nyS.append(run_time_ny)

    # plt.plot(samples_size_set,run_time_nyS)
    # plt.plot(samples_size_set,run_time_kerS)
    # plt.xlabel("number of samples selected")
    # plt.ylabel("running time")

    plt.plot(samples_size_set, acc_list)
    plt.ylabel("kernel approximation accuracy as in MAE")
    plt.xlabel("number of samples selected")
    plt.show()
    # ker_diff = np.abs(nystrom_output-kernel_output)
    # print(ker_diff)
    # print(sum(ker_diff))
