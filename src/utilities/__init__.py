import inspect
import numpy as np
import sys
import bottleneck

def instance_convert(X):
    if inspect.isclass(X):
        return X()
    return X


def sign(x):
    """ Returns sign. Numpys sign function returns 0 instead of 1 for zero values. :( """
    if x >= 0:
        return 1
    else:
        return -1

def getLc1(slf, H, gamma, q, i):
    # weird hacks below
    Lc1 = np.nan
    if gamma.size < 2:
        g = gamma
    else:
        g = gamma.item(i)

    if g <= 0:
        Lc1 = np.array(q * np.inf)
    elif (slf.weights[i]>0) or (slf.weights[i]==0 and H[i]<0):
        Lc1 = (-H[i] - slf.eps) / g
    else:
        Lc1 = (-H[i] + slf.eps) / g

    if np.isnan(Lc1):
        Lc1 = np.array(q * np.inf)
    return Lc1.item()

def getLc2(slf, H, q, i):
    if len(slf.supportSetIndices) > 0:
        if q > 0:
            Lc2 = -slf.weights[i] + slf.C
        else:
            Lc2 = -slf.weights[i] - slf.C
    else:
        Lc2 = np.array(q * np.inf)
    if np.isnan(Lc2):
        Lc2 = np.array(q * np.inf)
    return Lc2

def getLs(slf, H, beta, q):
    if len(slf.supportSetIndices) > 0 and len(beta) > 0:
        Ls = np.zeros([len(slf.supportSetIndices), 1])
        supportWeights = slf.weights[slf.supportSetIndices]
        supportH = H[slf.supportSetIndices]
        for k in range(len(slf.supportSetIndices)):
            if q * beta[k + 1] == 0:
                Ls[k] = q * np.inf
            elif q * beta[k + 1] > 0:    #support in positive set
                if supportH[k] > 0:
                    if supportWeights[k] < -slf.C:
                        Ls[k] = (-supportWeights[k] - slf.C) / beta[k + 1]
                    elif supportWeights[k] <= 0:
                        Ls[k] = -supportWeights[k] / beta[k + 1]
                    else:
                        Ls[k] = q * np.inf
                else:
                    if supportWeights[k] < 0:
                        Ls[k] = -supportWeights[k] / beta[k + 1]
                    elif supportWeights[k] <= slf.C:
                        Ls[k] = (-supportWeights[k] + slf.C) / beta[k + 1]
                    else:
                        Ls[k] = q * np.inf
            else:   #support in negative set
                if supportH[k] > 0:
                    if supportWeights[k] > 0:
                        Ls[k] = -supportWeights[k] / beta[k + 1]
                    elif supportWeights[k] >= -slf.C:
                        Ls[k] = (-supportWeights[k] - slf.C) / beta[k + 1]
                    else:
                        Ls[k] = q * np.inf
                else:
                    if supportWeights[k] > slf.C:
                        Ls[k] = (-supportWeights[k] + slf.C) / beta[k + 1]
                    elif supportWeights[k] >= slf.C:
                        Ls[k] = -supportWeights[k] / beta[k + 1]
                    else:
                        Ls[k] = q * np.inf
    else:
        Ls = np.array([q * np.inf])

    # Correct for NaN
    Ls[np.isnan(Ls)] = q * np.inf
    if Ls.size > 1:
        Ls.shape = (len(Ls), 1)
        # Check for broken signs
        for val in Ls:
            if sign(val) == -sign(q) and val != 0:
                print('Sign mismatch error in Ls! Exiting.')
                sys.exit()
    # print('findVarLs',Ls)
    return Ls

def getLe(slf, H, gamma, q):
    if len(slf.errorSetIndices) > 0:
        Le = np.zeros([len(slf.errorSetIndices), 1])
        errorGamma = gamma[slf.errorSetIndices]
        errorWeights = slf.weights[slf.errorSetIndices]
        errorH = H[slf.errorSetIndices]
        for k in range(len(slf.errorSetIndices)):
            if q * errorGamma[k] == 0:
                Le[k] = q * np.inf
            elif q * errorGamma[k] > 0:
                if errorWeights[k] > 0:
                    if errorH[k] < -slf.eps:
                        Le[k] = (-errorH[k] - slf.eps) / errorGamma[k]
                    else:
                        Le[k] = q * np.inf
                else:
                    if errorH[k] < slf.eps:
                        Le[k] = (-errorH[k] + slf.eps) / errorGamma[k]
                    else:
                        Le[k] = q * np.inf
            else:
                if errorWeights[k] > 0:
                    if errorH[k] > -slf.eps:
                        Le[k] = (-errorH[k] - slf.eps) / errorGamma[k]
                    else:
                        Le[k] = q * np.inf
                else:
                    if errorH[k] > slf.eps:
                        Le[k] = (-errorH[k] + slf.eps) / errorGamma[k]
                    else:
                        Le[k] = q * np.inf
    else:
        Le = np.array([q * np.inf])

    # Correct for NaN
    Le[np.isnan(Le)] = q * np.inf
    if Le.size > 1:
        Le.shape = (len(Le), 1)
        # Check for broken signs
        for val in Le:
            if sign(val) == -sign(q) and val != 0:
                print('Sign mismatch error in Le! Exiting.')
                sys.exit()
    # print('findVarLe',Le)
    return Le

def getLr(slf, H, gamma, q):
    if len(slf.remainderSetIndices) > 0:
        Lr = np.zeros([len(slf.remainderSetIndices), 1])
        remGamma = gamma[slf.remainderSetIndices]
        remH = H[slf.remainderSetIndices]
        for k in range(len(slf.remainderSetIndices)):
            if q * remGamma[k] == 0:
                Lr[k] = q * np.inf
            elif q * remGamma[k] > 0:
                if remH[k] < -slf.eps:
                    Lr[k] = (-remH[k] - slf.eps) / remGamma[k]
                elif remH[k] < slf.eps:
                    Lr[k] = (-remH[k] + slf.eps) / remGamma[k]
                else:
                    Lr[k] = q * np.inf
            else:
                if remH[k] > slf.eps:
                    Lr[k] = (-remH[k] + slf.eps) / remGamma[k]
                elif remH[k] > -slf.eps:
                    Lr[k] = (-remH[k] - slf.eps) / remGamma[k]
                else:
                    Lr[k] = q * np.inf
    else:
        Lr = np.array([q * np.inf])

    # Correct for NaN
    Lr[np.isnan(Lr)] = q * np.inf
    if Lr.size > 1:
        Lr.shape = (len(Lr), 1)
        # Check for broken signs
        for val in Lr:
            if sign(val) == -sign(q) and val != 0:
                print('Sign mismatch error in Lr! Exiting.')
                sys.exit()
    # print('findVarLr',Lr)
    return Lr

def getNMinSupport(slf, H, beta, gamma, i,kn):
    # Find direction q of the new sample
    if len(slf.supportSetIndices) <= (kn + 1):
        return []
    number_less_significant = int(len(slf.supportSetIndices) / (kn + 1))
    # number_less_significant = (kn + 1)
    q = -sign(H[i])
    Lc1 = getLc1(slf, H, gamma, q, i)
    q = sign(Lc1)
    if len(beta) < len(slf.supportSetIndices):
        Ls = getLs(slf, H, beta, q)
    else:
        Ls = getLs(slf, H, np.vstack((beta,slf.weights[-1])), q)

    # Support set
    # n_inf = np.count_nonzero(np.isinf(np.abs(Ls)[:, 0]))
    if Ls.size >= number_less_significant:
        minS = bottleneck.argpartition(-np.abs(Ls)[:,0], 0)[:number_less_significant]
        minIS = [item for item in minS if item in slf.supportSetIndices]
                 # and not (np.abs(Ls[item]) == np.inf)]

        minLsIndex = [slf.supportSetIndices[i] for i in minIS]
        return minLsIndex
    minLsIndex = np.abs(Ls).argsort()[:2]
    return minLsIndex

def getMinVariation(slf, H, beta, gamma, i):
    # Find direction q of the new sample
    q = -sign(H[i])
    Lc1 = getLc1(slf, H, gamma, q, i)
    q = sign(Lc1)
    Lc2 = getLc2(slf, H, q, i)
    Ls = getLs(slf, H, beta, q)
    Le = getLe(slf, H, gamma, q)
    Lr = getLr(slf, H, gamma, q)

    # Support set
    if Ls.size > 1:
        minS = np.abs(Ls).min()
        results = np.array([k for k, val in enumerate(Ls)
                            if np.abs(val) == minS])
        if len(results) > 1:
            betaIndex = beta[results + 1].argmax()
            Ls[results] = q * np.inf
            Ls[results[betaIndex]] = q * minS
    # Error set
    if Le.size > 1:
        minE = np.abs(Le).min()
        results = np.array([k for k, val in enumerate(Le)
                            if np.abs(val) == minE])
        if len(results) > 1:
            errorGamma = gamma[slf.errorSetIndices]
            gammaIndex = errorGamma[results].argmax()
            Le[results] = q * np.inf
            Le[results[gammaIndex]] = q * minE
    # Remainder Set
    if Lr.size > 1:
        minR = np.abs(Lr).min()
        results = np.array([k for k, val in enumerate(Lr)
                            if np.abs(val) == minR])
        if len(results) > 1:
            remGamma = gamma[slf.remainderSetIndices]
            gammaIndex = remGamma[results].argmax()
            Lr[results] = q * np.inf
            Lr[results[gammaIndex]] = q * minR

    # Find minimum absolute variation of all, retain signs. Flag determines set-switching cases.
    minLsIndex = np.abs(Ls).argmin()
    minLeIndex = np.abs(Le).argmin()
    minLrIndex = np.abs(Lr).argmin()
    minIndices = [None, None, minLsIndex, minLeIndex, minLrIndex]
    minValues = np.array([Lc1, Lc2, Ls[minLsIndex],
                          Le[minLeIndex], Lr[minLrIndex]])

    if np.abs(minValues).min() == np.inf:
        print('No weights to modify! Something is wrong.')
        sys.exit()
    flag = np.abs(minValues).argmin()
    if slf.debug:
        print('MinValues', minValues)
    return minValues[flag], flag, minIndices[flag]