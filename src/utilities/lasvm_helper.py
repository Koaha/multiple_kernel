import numpy as np


class LaSVMHelper:
    """
    LaSVMHelper provides supporting functions for LaSVM optimization,
    including efficient computation of gradient, Lagrange multipliers,
    and adjustments for support vectors using vectorized operations.

    Methods
    -------
    instance_convert(X)
        Converts a callable class instance if necessary.

    sign(x)
        Returns the sign of a value, where 0 is treated as 1.

    getLc1(slf, H, gamma, q, i)
        Computes Lagrange multiplier bound Lc1 for support vector.

    getLc2(slf, H, q, i)
        Computes Lagrange multiplier bound Lc2 for alternative direction.

    getLs(slf, H, beta, q)
        Determines minimum variation needed for support vector update.

    getLe(slf, H, gamma, q)
        Determines minimum variation needed for error set update.

    getLr(slf, H, gamma, q)
        Determines minimum variation needed for remainder set update.

    getNMinSupport(slf, H, beta, gamma, i, kn)
        Identifies least significant support vectors based on criteria.

    getMinVariation(slf, H, beta, gamma, i)
        Determines minimum update required across all sets.
    """

    @staticmethod
    def instance_convert(X):
        if callable(X):
            return X()
        return X

    @staticmethod
    def sign(x):
        """Efficient sign function that returns 1 for zero values."""
        return np.sign(x) if x != 0 else 1

    @staticmethod
    def getLc1(slf, H, gamma, q, i):
        g = gamma[i] if gamma.size >= 2 else gamma
        if g <= 0:
            return q * np.inf

        adjustment = (
            (-H[i] - slf.eps) / g
            if (slf.weights[i] > 0) or (H[i] < 0)
            else (-H[i] + slf.eps) / g
        )
        return adjustment if not np.isnan(adjustment) else q * np.inf

    @staticmethod
    def getLc2(slf, H, q, i):
        if slf.supportSetIndices:
            return -slf.weights[i] + slf.C if q > 0 else -slf.weights[i] - slf.C
        return q * np.inf

    @staticmethod
    def getLs(slf, H, beta, q):
        if not slf.supportSetIndices or beta.size == 0:
            return np.full((1,), q * np.inf)

        supportH = H[slf.supportSetIndices]
        supportWeights = slf.weights[slf.supportSetIndices]
        beta_k = beta[1:]

        # Vectorized computation for Ls
        Ls = np.where(
            (q * beta_k == 0),
            q * np.inf,
            np.where(
                (q * beta_k > 0) & (supportH > 0),
                np.where(
                    (supportWeights < -slf.C),
                    (-supportWeights - slf.C) / beta_k,
                    np.where(supportWeights <= 0, -supportWeights / beta_k, q * np.inf),
                ),
                np.where(
                    (supportH <= 0) & (supportWeights >= 0),
                    (-supportWeights + slf.C) / beta_k,
                    q * np.inf,
                ),
            ),
        )

        return np.nan_to_num(Ls, nan=q * np.inf).reshape(-1, 1)

    @staticmethod
    def getLe(slf, H, gamma, q):
        if not slf.errorSetIndices:
            return np.full((1,), q * np.inf)

        errorGamma = gamma[slf.errorSetIndices]
        errorH = H[slf.errorSetIndices]

        # Vectorized computation for Le
        Le = np.where(
            (q * errorGamma == 0),
            q * np.inf,
            np.where(
                (q * errorGamma > 0) & (errorH < -slf.eps),
                (-errorH - slf.eps) / errorGamma,
                np.where(
                    (q * errorGamma < 0) & (errorH > slf.eps),
                    (-errorH + slf.eps) / errorGamma,
                    q * np.inf,
                ),
            ),
        )

        return np.nan_to_num(Le, nan=q * np.inf).reshape(-1, 1)

    @staticmethod
    def getLr(slf, H, gamma, q):
        if not slf.remainderSetIndices:
            return np.full((1,), q * np.inf)

        remGamma = gamma[slf.remainderSetIndices]
        remH = H[slf.remainderSetIndices]

        # Vectorized computation for Lr
        Lr = np.where(
            (q * remGamma == 0),
            q * np.inf,
            np.where(
                (q * remGamma > 0) & (remH < -slf.eps),
                (-remH - slf.eps) / remGamma,
                np.where(
                    (q * remGamma < 0) & (remH > slf.eps),
                    (-remH + slf.eps) / remGamma,
                    q * np.inf,
                ),
            ),
        )

        return np.nan_to_num(Lr, nan=q * np.inf).reshape(-1, 1)

    @staticmethod
    def getNMinSupport(slf, H, beta, gamma, i, kn=4):
        if len(slf.supportSetIndices) <= kn + 1:
            return []

        num_least_significant = len(slf.supportSetIndices) // (kn + 1)
        q = -LaSVMHelper.sign(H[i])
        Ls = LaSVMHelper.getLs(slf, H, beta, q)

        min_indices = np.argsort(np.abs(Ls), axis=0)[:num_least_significant].flatten()
        return [
            slf.supportSetIndices[idx]
            for idx in min_indices
            if idx in slf.supportSetIndices
        ]

    @staticmethod
    def getMinVariation(slf, H, beta, gamma, i):
        q = -LaSVMHelper.sign(H[i])
        Lc1 = LaSVMHelper.getLc1(slf, H, gamma, q, i)
        q = LaSVMHelper.sign(Lc1)

        Lc2 = LaSVMHelper.getLc2(slf, H, q, i)
        Ls = LaSVMHelper.getLs(slf, H, beta, q)
        Le = LaSVMHelper.getLe(slf, H, gamma, q)
        Lr = LaSVMHelper.getLr(slf, H, gamma, q)

        min_values = np.array(
            [Lc1, Lc2, np.min(np.abs(Ls)), np.min(np.abs(Le)), np.min(np.abs(Lr))]
        )
        flag = np.argmin(np.abs(min_values))

        if np.isinf(min_values[flag]):
            raise RuntimeError("No weights to modify, convergence issue.")

        min_indices = [
            None,
            None,
            np.argmin(np.abs(Ls)),
            np.argmin(np.abs(Le)),
            np.argmin(np.abs(Lr)),
        ]
        return min_values[flag], flag, min_indices[flag]
