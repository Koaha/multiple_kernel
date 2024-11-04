from collections import defaultdict
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from metrics.metricsCalculator import (
    MetricsCalculator,
)  # Assuming this is the correct import path


class Validation:
    """
    Validation helper class to perform various validation strategies on SVM models
    and compute evaluation metrics using a custom MetricsCalculator.

    Parameters
    ----------
    model : object
        The SVM model instance that supports fit and predict methods.
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features).
    Y : np.ndarray
        Labels, shape (n_samples,).
    validation_type : str, optional, default='k_fold'
        Validation strategy to use. Options include:
        - 'k_fold': K-fold cross-validation.
        - 'stratified_k_fold': Stratified K-fold cross-validation.
        - 'hold_out': Single split (train/test) validation.
        - 'random_split': Random splits for validation (repeat multiple times).
    k_folds : int, optional, default=5
        Number of folds for cross-validation (used in 'k_fold' and 'stratified_k_fold').
    test_size : float, optional, default=0.2
        Proportion of the data to use as the test set in 'hold_out' and 'random_split'.
    n_splits : int, optional, default=10
        Number of times to perform random split validation if using 'random_split'.
    metrics : list of str, optional, default=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        Metrics to compute. Options are based on custom MetricsCalculator options.

    Attributes
    ----------
    results : dict
        Dictionary containing average values of each metric across splits.
    split_results : list of dict
        Detailed metrics for each split or fold.
    """

    def __init__(
        self,
        model,
        X,
        Y,
        validation_type="k_fold",
        k_folds=5,
        test_size=0.2,
        n_splits=10,
        metrics=["accuracy", "precision", "recall", "f1", "roc_auc"],
    ):
        self.model = model
        self.X = X
        self.Y = Y
        self.validation_type = validation_type
        self.k_folds = k_folds
        self.test_size = test_size
        self.n_splits = n_splits
        self.metrics = metrics
        self.results = {}
        self.split_results = []
        self.metrics_calculator = MetricsCalculator(
            metrics=self.metrics
        )  # Initialize custom metrics calculator

    def evaluate(self):
        """
        Perform validation based on the specified validation type and calculate metrics.

        Returns
        -------
        dict
            Average values of each metric across splits.
        """
        if self.validation_type == "k_fold":
            return self.k_fold_cross_validation()
        elif self.validation_type == "stratified_k_fold":
            return self.stratified_k_fold_cross_validation()
        elif self.validation_type == "hold_out":
            return self.hold_out_validation()
        elif self.validation_type == "random_split":
            return self.random_split_validation()
        else:
            raise ValueError(f"Unsupported validation type '{self.validation_type}'")

    def k_fold_cross_validation(self):
        """
        Performs K-fold cross-validation.

        Returns
        -------
        dict
            Average metrics across folds.
        """
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        return self._execute_splits(kf.split(self.X))

    def stratified_k_fold_cross_validation(self):
        """
        Performs Stratified K-fold cross-validation to ensure balanced class distribution.

        Returns
        -------
        dict
            Average metrics across stratified folds.
        """
        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        return self._execute_splits(skf.split(self.X, self.Y))

    def hold_out_validation(self):
        """
        Performs single hold-out validation, splitting the data into train and test sets.

        Returns
        -------
        dict
            Metrics for the single hold-out validation split.
        """
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X, self.Y, test_size=self.test_size, random_state=42
        )
        return self._train_evaluate_split(X_train, X_test, Y_train, Y_test)

    def random_split_validation(self):
        """
        Performs random split validation multiple times and averages the results.

        Returns
        -------
        dict
            Average metrics across random splits.
        """
        metrics_tracker = defaultdict(list)

        for _ in range(self.n_splits):
            X_train, X_test, Y_train, Y_test = train_test_split(
                self.X,
                self.Y,
                test_size=self.test_size,
                random_state=np.random.randint(10000),
            )
            fold_metrics = self._train_evaluate_split(X_train, X_test, Y_train, Y_test)
            self.split_results.append(fold_metrics)

            for metric, value in fold_metrics.items():
                metrics_tracker[metric].append(value)

        self.results = {
            metric: np.mean(values) for metric, values in metrics_tracker.items()
        }
        return self.results

    def _execute_splits(self, split_indices):
        """
        Executes model training and evaluation for each split in the cross-validation.

        Parameters
        ----------
        split_indices : generator
            Indices for train and test sets for each split.

        Returns
        -------
        dict
            Average metrics across all splits.
        """
        metrics_tracker = defaultdict(list)

        for train_index, test_index in split_indices:
            X_train, X_test = self.X[train_index], self.X[test_index]
            Y_train, Y_test = self.Y[train_index], self.Y[test_index]

            fold_metrics = self._train_evaluate_split(X_train, X_test, Y_train, Y_test)
            self.split_results.append(fold_metrics)

            for metric, value in fold_metrics.items():
                metrics_tracker[metric].append(value)

        self.results = {
            metric: np.mean(values) for metric, values in metrics_tracker.items()
        }
        return self.results

    def _train_evaluate_split(self, X_train, X_test, Y_train, Y_test):
        """
        Train the model on the training set and evaluate on the test set using custom metrics.

        Parameters
        ----------
        X_train : np.ndarray
            Training data.
        X_test : np.ndarray
            Test data.
        Y_train : np.ndarray
            Training labels.
        Y_test : np.ndarray
            Test labels.

        Returns
        -------
        dict
            Computed metrics for the split.
        """
        self.model.fit(X_train, Y_train)
        predictions = self.model.predict(X_test)
        return self.metrics_calculator.compute_metrics(Y_test, predictions)

    def detailed_split_results(self):
        """
        Get detailed metrics for each split or fold.

        Returns
        -------
        list of dict
            List containing metrics for each split or fold.
        """
        return self.split_results


# Example usage
# -------------
# Assuming `model` is an instance of a compatible SVM model and MetricsCalculator supports relevant metrics:
# >>> validator = Validation(model, X, Y, validation_type='stratified_k_fold', k_folds=5)
# >>> results = validator.evaluate()
# >>> print("Average validation metrics:", results)
# >>> print("Detailed split metrics:", validator.detailed_split_results())
