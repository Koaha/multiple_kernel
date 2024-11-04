# optimization_helper.py

import numpy as np


class OptimizationHelper:
    """
    A helper class for optimization-related calculations.

    Methods
    -------
    get_min_variation(H, beta, gamma, index)
        Determines the minimum variation for the sample update step.

    determine_flag(variation, margin)
        Determines the adjustment flag based on the variation and margin.
    """

    def get_min_variation(self, H, beta, gamma, index):
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
        float
            Minimum variation in weights.

        int
            Adjustment flag for set management.

        int
            Index of the minimum-variation sample.
        """
        variations = H - gamma[index] * beta
        min_variation = np.min(variations)
        minIndex = np.argmin(variations)
        flag = self.determine_flag(min_variation, H[index])
        deltaC = min_variation - H[index]
        return deltaC, flag, minIndex

    def determine_flag(self, variation, margin):
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
            Adjustment flag based on the margin variation.
        """
        if margin > 0 and variation < 0:
            return 0  # Add to support
        elif margin < 0 and variation < 0:
            return 1  # Add to error
        elif margin > 0:
            return 3  # Move to error
        else:
            return 4  # Move to remainder
