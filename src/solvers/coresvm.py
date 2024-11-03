import numpy as np
from src.kernels.kernels import RBF, Linear, Polynomial
from src.coresvm.smo_solver import SMO
from src.coresvm.pegasos_solver import Pegasos
from src.coresvm.cutting_plane import CP

class CoreSVM:
    """
    CoreSVM is a custom Support Vector Machine (SVM) classifier supporting multiple solvers
    (Cutting Plane, SMO, Pegasos) and various kernel functions (RBF, Linear, Polynomial).
    
    Parameters
    ----------
    X : np.ndarray
        Training data, shape (n_samples, n_features).
    Y : np.ndarray
        Training labels, shape (n_samples,).
    solver : str, optional, default='smo'
        Solver type to be used in the algorithm. It must be one of:
        'cutting_plane', 'smo', 'pegasos'.
    kernel : str, optional, default='RBF'
        Kernel type to be used in the algorithm. It must be one of:
        'RBF', 'Linear', 'Polynomial'.
    C : float, optional, default=1.0
        Regularization parameter. The strength of the regularization 
        is inversely proportional to C. Must be strictly positive.
    max_iter : int, optional, default=1000
        Number of iterations for training.
    mode : str, optional, default='primal'
        Mode for the Cutting Plane solver, either 'primal' or 'dual'.
    
    Attributes
    ----------
    model : CP, SMO, or Pegasos
        The initialized solver model used for training and prediction.
    """

    def __init__(self, X, Y, solver="smo", kernel="RBF", C=1.0, max_iter=1000, mode="primal"):
        self.X = X
        self.Y = Y
        self.C = C
        self.max_iter = max_iter
        self.mode = mode
        self.kernel_name = kernel
        self.solver_type = solver

        # Validate parameters
        if C <= 0:
            raise ValueError("Regularization parameter C must be positive.")
        if max_iter <= 0:
            raise ValueError("Number of iterations must be positive.")
        if X.shape[0] != len(Y):
            raise ValueError("Mismatch in the number of samples between X and Y.")

        # Initialize kernel
        kernel_dict = {
            'RBF': RBF(),
            'Linear': Linear(),
            'Polynomial': Polynomial()
        }
        self.kernel = kernel_dict.get(kernel, kernel_dict['RBF'])

        # Initialize solver based on user choice
        if solver == "cutting_plane":
            self.model = CP(mode=self.mode)
        elif solver == "smo":
            self.model = SMO(X, Y, C=self.C, tol=1e-4, max_iter=self.max_iter, kernel=self.kernel)
        elif solver == "pegasos":
            self.model = Pegasos(X, Y, lambda_param=1 / C, max_iter=self.max_iter, kernel=self.kernel)
        else:
            raise ValueError(f"Unsupported solver type '{solver}'. Choose from 'cutting_plane', 'smo', 'pegasos'.")

    def fit(self):
        """
        Train the SVM model using the specified solver and mode.

        For Cutting Plane, trains using either the primal or dual formulation based on mode.
        For SMO and Pegasos, trains using the standard SVM dual formulation.
        """
        if self.solver_type == "cutting_plane":
            if self.mode == "dual":
                self.model.fit(self.X, self.Y, C=self.C, epsilon=1e-4, s=min(10, len(self.Y)), kernel_func=self.kernel)
            else:
                self.model.fit(self.X, self.Y, C=self.C, epsilon=1e-4)
        elif self.solver_type == "smo":
            self.model.fit()
        elif self.solver_type == "pegasos":
            self.model.fit()

    def predict(self, X):
        """
        Predict the class labels for samples in X.

        Parameters
        ----------
        X : np.ndarray
            Data to be classified, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted class labels, shape (n_samples,).
        """
        if self.solver_type == "cutting_plane":
            if self.mode == "dual":
                return self.model.predict(X, kernel_func=self.kernel)
            else:
                return self.model.predict(X)
        elif self.solver_type in ["smo", "pegasos"]:
            return self.model.predict(X)

    def score(self, X, Y):
        """
        Calculate the accuracy of the classifier on the test data.

        Parameters
        ----------
        X : np.ndarray
            Test data, shape (n_samples, n_features).
        Y : np.ndarray
            True labels, shape (n_samples,).

        Returns
        -------
        float
            Accuracy score as the fraction of correct predictions.
        """
        predictions = self.predict(X)
        return np.mean(predictions == Y)

    def decision_function(self, X):
        """
        Computes the decision function for each sample in X.

        Parameters
        ----------
        X : np.ndarray
            Data points to evaluate, shape (n_samples, n_features).
        
        Returns
        -------
        np.ndarray
            Decision function values, shape (n_samples,).
        """
        return self.model.decision_function(X)
