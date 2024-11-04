import numpy as np
from src.kernels.kernels import RBF, Linear, Polynomial
from src.coresvm.smo_solver import SMO
from src.coresvm.pegasos_solver import Pegasos
from src.coresvm.cutting_plane import CP
from src.metrics.metrics_tracker import (
    MetricsTracker,
)  # Ensure MetricsTracker is imported correctly


class CoreSVM:
    """
    CoreSVM is a custom Support Vector Machine (SVM) classifier supporting multiple solvers
    (Cutting Plane, SMO, Pegasos) and various kernel functions (RBF, Linear, Polynomial).
    It integrates a real-time metrics tracker for online evaluation during training.

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
    enable_metrics : bool, optional, default=True
        Enables real-time metrics tracking during training if set to True.

    Attributes
    ----------
    model : CP, SMO, or Pegasos
        The initialized solver model used for training and prediction.
    metrics_tracker : MetricsTracker
        Instance for tracking and visualizing model metrics during training.

    Examples
    --------
    # Basic usage with real-time metrics enabled
    >>> X_train, Y_train = np.array([[1, 2], [2, 3], [3, 4]]), np.array([1, -1, 1])
    >>> X_test, Y_test = np.array([[2, 3], [3, 5]]), np.array([1, -1])
    >>> svm = CoreSVM(X_train, Y_train, solver='smo', kernel='RBF', enable_metrics=True)
    >>> svm.fit()
    >>> accuracy = svm.score(X_test, Y_test)
    >>> print(f"Test accuracy: {accuracy}")
    """

    def __init__(
        self,
        X,
        Y,
        solver="smo",
        kernel="RBF",
        C=1.0,
        max_iter=1000,
        mode="primal",
        enable_metrics=True,
    ):
        self.X = X
        self.Y = Y
        self.C = C
        self.max_iter = max_iter
        self.mode = mode
        self.kernel_name = kernel
        self.solver_type = solver
        self.enable_metrics = enable_metrics

        # Validate parameters
        if C <= 0:
            raise ValueError("Regularization parameter C must be positive.")
        if max_iter <= 0:
            raise ValueError("Number of iterations must be positive.")
        if X.shape[0] != len(Y):
            raise ValueError("Mismatch in the number of samples between X and Y.")

        # Initialize kernel
        kernel_dict = {"RBF": RBF(), "Linear": Linear(), "Polynomial": Polynomial()}
        self.kernel = kernel_dict.get(kernel, kernel_dict["RBF"])

        # Initialize solver based on user choice
        if solver == "cutting_plane":
            self.model = CP(mode=self.mode)
        elif solver == "smo":
            self.model = SMO(
                X, Y, C=self.C, tol=1e-4, max_iter=self.max_iter, kernel=self.kernel
            )
        elif solver == "pegasos":
            self.model = Pegasos(
                X, Y, lambda_param=1 / C, max_iter=self.max_iter, kernel=self.kernel
            )
        else:
            raise ValueError(
                f"Unsupported solver type '{solver}'. Choose from 'cutting_plane', 'smo', 'pegasos'."
            )

        # Initialize the metrics tracker if enabled
        self.metrics_tracker = MetricsTracker() if self.enable_metrics else None

    def fit(self):
        """
        Train the SVM model using the specified solver and mode.

        During training, updates real-time metrics if enabled.

        For Cutting Plane, trains using either the primal or dual formulation based on mode.
        For SMO and Pegasos, trains using the standard SVM dual formulation.
        """
        if self.solver_type == "cutting_plane":
            if self.mode == "dual":
                self.model.fit(
                    self.X,
                    self.Y,
                    C=self.C,
                    epsilon=1e-4,
                    s=min(10, len(self.Y)),
                    kernel_func=self.kernel,
                )
            else:
                self.model.fit(self.X, self.Y, C=self.C, epsilon=1e-4)
        elif self.solver_type == "smo":
            self.model.fit()
        elif self.solver_type == "pegasos":
            self.model.fit()

        if self.enable_metrics:
            # Update metrics tracker with current accuracy after training
            training_accuracy = self.score(self.X, self.Y)
            self.metrics_tracker.update("accuracy", training_accuracy)

            # Finalize and plot the training metrics
            self.metrics_tracker.finalize()
            self.metrics_tracker.visualize_metrics()

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
        accuracy = np.mean(predictions == Y)

        if self.enable_metrics:
            # Track accuracy over time if real-time metrics are enabled
            self.metrics_tracker.update("accuracy", accuracy)

        return accuracy

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
