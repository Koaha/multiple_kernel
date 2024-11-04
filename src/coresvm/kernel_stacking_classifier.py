from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from src.solvers.coresvm import CoreSVM
from src.solvers.nysvm import NySVM
from src.solvers.lasvm import LaSVM
from validation.validation import Validation
import numpy as np


class KernelStackingClassifier(BaseEstimator, ClassifierMixin):
    """
    Stacked classifier with out-of-fold meta-features, adaptive weighting, and meta-classifier selection.

    Parameters
    ----------
    C : float
        Regularization parameter for each SVM.
    base_estimators : list, optional
        List of base SVM estimators (default: [CoreSVM, NySVM, LaSVM] with different kernels).
    meta_classifiers : list, optional
        List of meta-classifiers to choose from (default: [LogisticRegression, RidgeClassifier, SGDClassifier]).
    n_folds : int, default=5
        Number of cross-validation folds.
    adaptive_weights : bool, optional
        If True, assigns weights to base models based on performance.
    validation_type : str, optional
        Type of validation to use. Options: 'k_fold', 'stratified_k_fold', 'hold_out', 'random_split'.
    metrics : list of str, optional
        Metrics to compute during validation.

    Attributes
    ----------
    weights : np.ndarray
        Weights assigned to each model based on validation scores.
    meta_classifier : object
        The selected meta-classifier for final predictions.

    Examples
    --------
    >>> stacking_clf = EnhancedKernelStackingClassifier(C=1.0, n_folds=5, adaptive_weights=True)
    >>> stacking_clf.fit(X_train, y_train)
    >>> stacking_clf.evaluate(X_train, y_train, validation_type='stratified_k_fold', metrics=['accuracy', 'precision'])
    >>> predictions = stacking_clf.predict(X_test)
    """

    def __init__(
        self,
        C=1.0,
        base_estimators=None,
        meta_classifiers=None,
        n_folds=5,
        adaptive_weights=True,
        validation_type="k_fold",
        metrics=["accuracy", "precision"],
    ):
        self.C = C
        self.base_estimators = base_estimators or [
            CoreSVM(C=self.C, kernel=Linear()),
            NySVM(C=self.C, kernel=RBF()),
            LaSVM(C=self.C, kernel=Polynomial()),
        ]
        self.meta_classifiers = meta_classifiers or [
            LogisticRegression(),
            RidgeClassifier(),
            SGDClassifier(),
        ]
        self.n_folds = n_folds
        self.adaptive_weights = adaptive_weights
        self.validation_type = validation_type
        self.metrics = metrics
        self.weights = np.ones(len(self.base_estimators)) / len(self.base_estimators)
        self.meta_classifier = None  # To be set after selection

    def fit(self, X, y):
        """
        Fit the base estimators and select the best meta-classifier using cross-validation.

        Parameters
        ----------
        X : np.ndarray
            Training data, shape (n_samples, n_features).
        y : np.ndarray
            Training labels, shape (n_samples,).

        Returns
        -------
        None
        """
        n_samples = X.shape[0]
        meta_features = np.zeros((n_samples, len(self.base_estimators)))

        skf = StratifiedKFold(n_splits=self.n_folds)
        for i, estimator in enumerate(self.base_estimators):
            oof_preds = cross_val_predict(estimator, X, y, cv=skf, method="predict")
            estimator.fit(X, y)
            meta_features[:, i] = oof_preds

            if self.adaptive_weights:
                # Compute adaptive weights for each estimator
                self.weights[i] = Validation(
                    estimator,
                    X,
                    y,
                    self.validation_type,
                    self.n_folds,
                    metrics=self.metrics,
                ).evaluate()

        # Normalize weights if adaptive weighting is used
        if self.adaptive_weights:
            self.weights /= self.weights.sum()

        best_score = -np.inf
        for clf in self.meta_classifiers:
            score = cross_val_predict(
                clf, meta_features, y, cv=skf, method="predict_proba"
            ).mean()
            if score > best_score:
                best_score = score
                self.meta_classifier = clf

        self.meta_classifier.fit(meta_features, y)

    def predict(self, X):
        """
        Predict class labels using stacked SVM predictions as input to the meta-classifier.

        Parameters
        ----------
        X : np.ndarray
            Data to predict, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted class labels, shape (n_samples,).
        """
        meta_features = np.column_stack(
            [
                estimator.predict(X) * weight
                for estimator, weight in zip(self.base_estimators, self.weights)
            ]
        )
        return self.meta_classifier.predict(meta_features)

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
