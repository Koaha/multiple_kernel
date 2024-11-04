import numpy as np
from src.kernels.kernels import *
from src.operators.operators import *


class KernelTree:
    """
    KernelTree dynamically constructs combinations of base kernels with operators to
    create complex, customized kernels. Supports configurable depth to control the
    complexity of kernel compositions.

    Attributes
    ----------
    depth : int
        Maximum depth of kernel compositions.
    operations : list
        List of available operators for combining kernels.
    base_kernels : list
        List of base kernel functions.
    constructed_kernels : dict
        Dictionary storing kernels constructed at each depth level.
    """

    def __init__(self, depth=2):
        self.depth = depth
        self.operations = [
            OperatorAffine(),
            OperatorMultiplication(),
            OperatorPolynomial(),
            OperatorExponential(),
            OperatorCosine(),
            OperatorMin(),
            OperatorMax(),
        ]
        self.base_kernels = [Linear(), Polynomial(), RBF(), Laplacian(), Sigmoid()]
        self.constructed_kernels = {1: self.base_kernels}

    def construct_children(self, parent_kernel):
        """
        Generates child kernels by applying each operator to the parent kernel
        with each of the base kernels, reducing redundant operations.

        Parameters
        ----------
        parent_kernel : callable
            The base kernel to which operators are applied.

        Returns
        -------
        list
            List of newly constructed kernels at the current depth.
        """
        children = []
        for kernel in self.base_kernels:
            for operation in self.operations:
                # Vectorized single-input or dual-input operations
                if isinstance(operation, (OperatorExponential, OperatorPolynomial)):
                    # Single input operator
                    child_kernel = lambda X, Y, op=operation: op(parent_kernel(X, Y))
                else:
                    # Dual input operator
                    child_kernel = lambda X, Y, op=operation: op(
                        parent_kernel(X, Y), kernel(X, Y)
                    )
                children.append(child_kernel)
        return children

    def construct_tree(self, current_depth=1):
        """
        Recursively constructs kernels up to the specified depth, with caching
        to avoid redundant calculations. Stops if maximum depth is reached.

        Parameters
        ----------
        current_depth : int, optional
            The current depth level, default is 1.
        """
        if current_depth >= self.depth:
            return

        current_level_kernels = []
        # Apply operations to kernels from the previous depth level only
        for parent_kernel in self.constructed_kernels.get(current_depth, []):
            current_level_kernels.extend(self.construct_children(parent_kernel))

        self.constructed_kernels[current_depth + 1] = current_level_kernels
        self.construct_tree(current_depth + 1)

    def get_all_kernels(self):
        """
        Retrieves all constructed kernels as a flattened list.

        Returns
        -------
        list
            List of all kernels constructed up to the specified depth.
        """
        return [
            kernel
            for kernel_list in self.constructed_kernels.values()
            for kernel in kernel_list
        ]

    def apply_kernel_chain(self, X, Y, weights=None):
        """
        Applies a chain of constructed kernels on data, optionally with weights, to
        combine multiple kernels into a single output matrix.

        Parameters
        ----------
        X : np.ndarray
            Input data array.
        Y : np.ndarray
            Input data array for pairwise comparison.
        weights : list, optional
            Weights for each kernel in the chain. Default is None (equal weights).

        Returns
        -------
        np.ndarray
            Combined kernel matrix for X and Y.
        """
        kernels = self.get_all_kernels()
        weights = weights or [1 / len(kernels)] * len(kernels)
        combined_kernel = np.zeros((X.shape[0], Y.shape[0]))

        # Vectorized application of kernels with weights
        for kernel, weight in zip(kernels, weights):
            combined_kernel += weight * kernel(X, Y)
        return combined_kernel
