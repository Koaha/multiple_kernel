from sklearn.base import BaseEstimator, ClassifierMixin
from src.kernels.kernels import RBF, Linear, Polynomial, Sigmoid
from src.solvers.lasvm import LaSVM
from validation.validation import Validation
import numpy as np
from scipy.stats import mode
from sklearn.utils import resample


class SVMEensemble(BaseEstimator, ClassifierMixin):
    """
    Enhanced Ensemble of SVMs with adaptive weighting, bagging, and cross-validation.

    Parameters
    ----------
    C : float
        Regularization parameter for each SVM.
    kernels : list, optional
        List of kernel instances to use in the ensemble (default: [RBF, Linear, Polynomial, Sigmoid]).
    voting : str, optional
        Voting method for ensemble. Options are 'hard' (majority) or 'soft' (average probabilities).
    adaptive_weights : bool, optional
        If True, assigns adaptive weights based on model performance.
    bagging : bool, optional
        If True, uses bootstrap aggregation for training each SVM on a subset of data.
    n_estimators : int, optional
        Number of models to use in the ensemble (only used if bagging=True).
    random_state : int, optional
        Seed for reproducibility (used for bagging).
    validation_type : str, optional
        Type of validation to use. Options: 'k_fold', 'stratified_k_fold', 'hold_out', 'random_split'.
    k_folds : int, optional
        Number of folds for K-fold cross-validation.
    metrics : list of str, optional
        Metrics to compute during validation.

    Attributes
    ----------
    models : list of SVM instances
        List of SVM models with different kernels.
    weights : np.ndarray
        Weights assigned to each model based on validation scores.

    Examples
    --------
    >>> ensemble = SVMEensemble(C=1.0, voting='hard', adaptive_weights=True)
    >>> ensemble.fit(X_train, y_train)
    >>> ensemble.evaluate(X_train, y_train, validation_type='k_fold', k_folds=5, metrics=['accuracy', 'precision'])
    >>> predictions = ensemble.predict(X_test)
    """

    def __init__(
        self,
        C=1.0,
        kernels=None,
        voting="hard",
        adaptive_weights=True,
        bagging=True,
        n_estimators=5,
        random_state=42,
        validation_type="k_fold",
        k_folds=5,
        metrics=["accuracy", "precision"],
    ):
        self.C = C
        self.kernels = kernels or [RBF(), Linear(), Polynomial(), Sigmoid()]
        self.voting = voting
        self.adaptive_weights = adaptive_weights
        self.bagging = bagging
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.validation_type = validation_type
        self.k_folds = k_folds
        self.metrics = metrics
        self.models = [LaSVM(C=self.C, kernel=kernel) for kernel in self.kernels]
        self.weights = np.ones(len(self.models)) / len(self.models)

    def fit(self, X, y):
        """
        Fit each SVM in the ensemble, potentially with bagging and adaptive weighting.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).
        y : np.ndarray
            Training labels of shape (n_samples,).

        Returns
        -------
        None
        """
        rng = np.random.default_rng(self.random_state)
        for i, model in enumerate(self.models):
            if self.bagging:
                X_sample, y_sample = resample(X, y, random_state=self.random_state + i)
            else:
                X_sample, y_sample = X, y
            model.fit(X_sample, y_sample)

            if self.adaptive_weights:
                # Calculate model performance to set adaptive weights
                self.weights[i] = Validation(
                    model,
                    X,
                    y,
                    self.validation_type,
                    self.k_folds,
                    metrics=self.metrics,
                ).evaluate()

        # Normalize weights if adaptive weighting is used
        if self.adaptive_weights:
            self.weights /= self.weights.sum()

    def predict(self, X):
        """
        Predict class labels using ensemble voting.

        Parameters
        ----------
        X : np.ndarray
            Data to predict, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted class labels, shape (n_samples,).
        """
        predictions = np.array([model.predict(X) for model in self.models])

        if self.voting == "hard":
            weighted_preds = np.dot(predictions.T, self.weights)
            return np.sign(weighted_preds)
        elif self.voting == "soft":
            return np.sign(predictions.T @ self.weights)
        else:
            raise ValueError("Invalid voting method. Choose 'hard' or 'soft'.")

    def evaluate(
        self,
        X,
        y,
        validation_type="k_fold",
        k_folds=5,
        metrics=["accuracy", "precision"],
    ):
        """
        Evaluate the model using cross-validation and specified metrics.

        Parameters
        ----------
        X : np.ndarray
            Validation data, shape (n_samples, n_features).
        y : np.ndarray
            True labels, shape (n_samples,).
        validation_type : str
            Type of validation to use. Options: 'k_fold', 'stratified_k_fold', 'hold_out', 'random_split'.
        k_folds : int
            Number of folds for K-fold cross-validation.
        metrics : list of str
            Metrics to calculate during validation.

        Returns
        -------
        dict
            Evaluation results for specified metrics.
        """
        validation = Validation(
            self,
            X,
            y,
            validation_type=validation_type,
            k_folds=k_folds,
            metrics=metrics,
        )
        return validation.evaluate()