# svm_wrapper.py

from solvers import CoreSVM, NySVM, LaSVM
from src.kernels.nystrom import NystromApproximation

class SVMWrapper:
    """
    SVMWrapper provides a unified interface to train and predict with different 
    types of SVM solvers: CoreSVM, NySVM, and LaSVM.

    Parameters
    ----------
    solver_type : str
        Specifies the type of solver to use. Options are 'coresvm', 'nysvm', and 'lasvm'.
    **kwargs : 
        Additional arguments passed to the selected solver.
    """
    
    def __init__(self, solver_type='coresvm', use_nystrom=False, nystrom_params=None, **kwargs):
        # Check if Nyström approximation is needed
        if use_nystrom:
            if nystrom_params is None:
                raise ValueError("Nyström parameters must be provided when use_nystrom is True.")
            nystrom = NystromApproximation(**nystrom_params)
            self.kernel_matrix = nystrom.compute_nystrom_approximation()
            kwargs['kernel'] = self.kernel_matrix
        else:
            self.kernel_matrix = None
        
        # Initialize the solver
        if solver_type == 'coresvm':
            self.model = CoreSVM(**kwargs)
        elif solver_type == 'nysvm':
            self.model = NySVM(**kwargs)
        elif solver_type == 'lasvm':
            self.model = LaSVM(**kwargs)
        else:
            raise ValueError("Invalid solver type. Choose from 'coresvm', 'nysvm', 'lasvm'.")
    
    def fit(self, X, Y):
        """
        Fits the SVM model to the provided data.

        Parameters
        ----------
        X : np.ndarray
            Training data, shape (n_samples, n_features).
        Y : np.ndarray
            Training labels, shape (n_samples,).
        """
        self.model.fit(X, Y)

    def predict(self, X):
        """
        Predicts the labels for the provided data.

        Parameters
        ----------
        X : np.ndarray
            Data to predict, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted class labels, shape (n_samples,).
        """
        return self.model.predict(X)

    def get_params(self):
        """
        Returns the parameters of the current solver.

        Returns
        -------
        dict
            Dictionary of parameters for the selected solver.
        """
        return self.model.get_params() if hasattr(self.model, "get_params") else {}

    def set_params(self, **params):
        """
        Sets the parameters of the current solver.

        Parameters
        ----------
        **params : dict
            Dictionary of parameters to set for the selected solver.
        """
        if hasattr(self.model, "set_params"):
            self.model.set_params(**params)
        else:
            raise NotImplementedError(f"{self.model.__class__.__name__} does not support parameter setting.")
