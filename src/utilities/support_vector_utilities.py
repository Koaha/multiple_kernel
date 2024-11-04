import numpy as np


class SupportVectorUtilities:
    """
    A utility class for managing support vector-related operations.

    Methods
    -------
    compute_margin(weights, X, Y, bias)
        Computes the margin values for each sample in X with labels Y.

    compute_gradient(S, weights, b)
        Computes the gradient for a support vector set S with weights.

    prune_support_vectors(weights, support_set_indices, threshold=1e-4)
        Prunes support vectors based on weight magnitude and a specified threshold.

    adjust_sets(H, weights, support_set_indices, error_set_indices, remainder_set_indices, flag, minIndex)
        Adjusts sets by moving samples between support, error, and remainder sets based on the specified flag.
    """

    def compute_margin(self, weights, X, Y, bias):
        """
        Computes margin values for each sample in X with labels Y.

        Parameters
        ----------
        weights : np.ndarray
            Weight vector for support vectors.

        X : np.ndarray
            Input matrix of shape (n_samples, n_features).

        Y : np.ndarray
            Target labels of shape (n_samples,).

        bias : float
            Bias term for margin calculation.

        Returns
        -------
        np.ndarray
            Margin values for each sample in X.
        """
        return (X.dot(weights) + bias) * Y

    def compute_gradient(self, S, weights, b):
        """
        Computes the gradient for a support vector set S with weights.

        Parameters
        ----------
        S : list of np.ndarray
            Support vector set.

        weights : np.ndarray
            Weights associated with support vectors.

        b : float
            Bias term.

        Returns
        -------
        np.ndarray
            Gradients for each vector in S.
        """
        return np.dot(S, weights) + b

    def prune_support_vectors(self, weights, support_set_indices, threshold=1e-4):
        """
        Prunes support vectors based on weight magnitude and threshold.

        Parameters
        ----------
        weights : np.ndarray
            Weight vector.

        support_set_indices : list of int
            Indices of current support vectors.

        threshold : float, optional (default=1e-4)
            Threshold below which support vectors are pruned.

        Returns
        -------
        list of int
            Updated support set indices with low-weight vectors removed.
        """
        return [i for i in support_set_indices if abs(weights[i]) >= threshold]

    def adjust_sets(
        self,
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
            Adjustment flag indicating the type of set adjustment.

        minIndex : int
            Index of the sample to adjust.

        Returns
        -------
        np.ndarray
            Updated margin array.
        """
        if flag == 0:  # Add to support set
            support_set_indices.append(minIndex)
            weights[minIndex] = np.sign(H[minIndex])
            H[minIndex] = weights[minIndex]
        elif flag == 1:  # Move to error set
            weights[minIndex] = np.sign(weights[minIndex]) * np.max(abs(weights))
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
