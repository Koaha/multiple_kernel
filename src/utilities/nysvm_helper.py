import numpy as np


class NySVMHelper:
    """
    NySVMHelper provides helper functions for the NySVM solver, enabling efficient
    margin calculation, gradient computation, support vector pruning, and set adjustment.

    Methods
    -------
    compute_margin(weights, X, Y, bias)
        Calculates the margin for each sample in X based on current weights and bias.

    compute_gradient(S, weights, b)
        Computes gradient values for support vector set S.

    prune_support_vectors(weights, support_set_indices, threshold)
        Prunes support vectors whose weights are below a specified threshold.

    adjust_sets(H, weights, support_set_indices, error_set_indices, remainder_set_indices, flag, minIndex)
        Adjusts support, error, and remainder sets based on optimization flags.

    get_min_variation(H, beta, gamma, index)
        Determines the minimum weight variation needed to update support vectors.
    """

    @staticmethod
    def compute_margin(weights, X, Y, bias):
        """
        Computes margin values for each sample in X with labels Y.

        Parameters
        ----------
        weights : np.ndarray
            Weight vector for support vectors.
        X : np.ndarray
            Input matrix, shape (n_samples, n_features).
        Y : np.ndarray
            Target labels, shape (n_samples,).
        bias : float
            Bias term for margin calculation.

        Returns
        -------
        np.ndarray
            Margin values for each sample in X.
        """
        return (X.dot(weights) + bias) * Y

    @staticmethod
    def compute_gradient(S, weights, b):
        """
        Computes the gradient for a set of support vectors.

        Parameters
        ----------
        S : np.ndarray
            Support vector set, shape (n_support_vectors, n_features).
        weights : np.ndarray
            Weights associated with support vectors.
        b : float
            Bias term.

        Returns
        -------
        np.ndarray
            Gradient values for each vector in S.
        """
        return S.dot(weights) + b

    @staticmethod
    def prune_support_vectors(weights, support_set_indices, threshold=1e-4):
        """
        Prunes support vectors with weights below a certain threshold.

        Parameters
        ----------
        weights : np.ndarray
            Weight vector.
        support_set_indices : list of int
            Indices of current support vectors.
        threshold : float, optional, default=1e-4
            Threshold below which support vectors are pruned.

        Returns
        -------
        list of int
            Updated list of support set indices with low-weight vectors removed.
        """
        return [i for i in support_set_indices if abs(weights[i]) >= threshold]

    @staticmethod
    def adjust_sets(
        H,
        weights,
        support_set_indices,
        error_set_indices,
        remainder_set_indices,
        flag,
        minIndex,
    ):
        """
        Adjusts sets by moving samples between support, error, and remainder sets.

        Parameters
        ----------
        H : np.ndarray
            Margin array.
        weights : np.ndarray
            Weight vector.
        support_set_indices : list of int
            Indices of support vectors.
        error_set_indices : list of int
            Indices of error vectors.
        remainder_set_indices : list of int
            Indices of remainder vectors.
        flag : int
            Adjustment flag indicating type of set adjustment.
        minIndex : int
            Index of the sample to adjust.

        Returns
        -------
        np.ndarray
            Updated margin array H.
        """
        if flag == 0:  # Add to support set
            support_set_indices.append(minIndex)
            weights[minIndex] = np.sign(H[minIndex])
            H[minIndex] = weights[minIndex]
        elif flag == 1:  # Move to error set
            weights[minIndex] = np.sign(weights[minIndex]) * max(abs(weights))
            error_set_indices.append(minIndex)
        elif flag == 2:  # Move from support to remainder
            support_set_indices.remove(minIndex)
            remainder_set_indices.append(minIndex)
        elif flag == 3:  # Move from error to support
            error_set_indices.remove(minIndex)
            support_set_indices.append(minIndex)
        elif flag == 4:  # Move from remainder to support
            remainder_set_indices.remove(minIndex)
            support_set_indices.append(minIndex)
        return H

    @staticmethod
    def get_min_variation(H, beta, gamma, index):
        """
        Determines the minimum variation for the sample update step.

        Parameters
        ----------
        H : np.ndarray
            Margin array.
        beta : np.ndarray
            Beta coefficients for margin adjustments.
        gamma : np.ndarray
            Gamma coefficients for margin adjustments.
        index : int
            Index of the sample to update.

        Returns
        -------
        deltaC : float
            Minimum variation in weights.
        flag : int
            Adjustment flag for set management.
        minIndex : int
            Index of the minimum-variation sample.
        """
        variations = H - gamma[index] * beta
        min_variation = np.min(variations)
        minIndex = np.argmin(variations)
        flag = NySVMHelper.determine_flag(min_variation, H[index])
        deltaC = min_variation - H[index]
        return deltaC, flag, minIndex

    @staticmethod
    def determine_flag(variation, margin):
        """
        Determines the adjustment flag based on margin variation.

        Parameters
        ----------
        variation : float
            Variation in margin.
        margin : float
            Margin value.

        Returns
        -------
        int
            Adjustment flag.
        """
        if margin > 0 and variation < 0:
            return 0  # Add to support
        elif margin < 0 and variation < 0:
            return 1  # Add to error
        elif margin > 0:
            return 3  # Move to error
        else:
            return 4  # Move to remainder
