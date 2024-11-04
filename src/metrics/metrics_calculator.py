import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
)


class MetricsCalculator:
    """
    A utility class for calculating evaluation metrics for classification and regression models.
    Supports standard metrics and custom metric functions.

    Parameters
    ----------
    custom_metrics : dict, optional
        Dictionary of custom metric functions. Each function should take y_true and y_pred as parameters.
        Format: {"metric_name": function}

    Examples
    --------
    # Using default metrics
    >>> calc = MetricsCalculator()
    >>> y_true = [1, 0, 1, 1]
    >>> y_pred = [1, 0, 0, 1]
    >>> calc.accuracy(y_true, y_pred)
    0.75

    # Adding custom metric
    >>> def custom_metric(y_true, y_pred):
    >>>     return np.mean(y_true == y_pred) * 100
    >>> calc = MetricsCalculator(custom_metrics={"custom_acc": custom_metric})
    >>> calc.custom_acc(y_true, y_pred)
    75.0
    """

    def __init__(self, custom_metrics=None):
        self.custom_metrics = custom_metrics or {}

    @staticmethod
    def accuracy(y_true, y_pred):
        """
        Calculates the accuracy score.

        Formula
        -------
        accuracy = (number of correct predictions) / (total predictions)

        Parameters
        ----------
        y_true : list or np.ndarray
            True labels.
        y_pred : list or np.ndarray
            Predicted labels.

        Returns
        -------
        float
            Accuracy score.

        Example
        -------
        >>> y_true = [1, 0, 1, 1]
        >>> y_pred = [1, 0, 0, 1]
        >>> MetricsCalculator.accuracy(y_true, y_pred)
        0.75
        """
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def precision(y_true, y_pred):
        """
        Calculates the precision score.

        Formula
        -------
        precision = (true positives) / (true positives + false positives)

        Parameters
        ----------
        y_true : list or np.ndarray
            True labels.
        y_pred : list or np.ndarray
            Predicted labels.

        Returns
        -------
        float
            Precision score.

        Example
        -------
        >>> y_true = [1, 0, 1, 1]
        >>> y_pred = [1, 0, 0, 1]
        >>> MetricsCalculator.precision(y_true, y_pred)
        0.67
        """
        return precision_score(y_true, y_pred)

    @staticmethod
    def recall(y_true, y_pred):
        """
        Calculates the recall score.

        Formula
        -------
        recall = (true positives) / (true positives + false negatives)

        Parameters
        ----------
        y_true : list or np.ndarray
            True labels.
        y_pred : list or np.ndarray
            Predicted labels.

        Returns
        -------
        float
            Recall score.

        Example
        -------
        >>> y_true = [1, 0, 1, 1]
        >>> y_pred = [1, 0, 0, 1]
        >>> MetricsCalculator.recall(y_true, y_pred)
        0.67
        """
        return recall_score(y_true, y_pred)

    @staticmethod
    def f1_score(y_true, y_pred):
        """
        Calculates the F1 score.

        Formula
        -------
        f1 = 2 * (precision * recall) / (precision + recall)

        Parameters
        ----------
        y_true : list or np.ndarray
            True labels.
        y_pred : list or np.ndarray
            Predicted labels.

        Returns
        -------
        float
            F1 score.

        Example
        -------
        >>> y_true = [1, 0, 1, 1]
        >>> y_pred = [1, 0, 0, 1]
        >>> MetricsCalculator.f1_score(y_true, y_pred)
        0.67
        """
        return f1_score(y_true, y_pred)

    @staticmethod
    def roc_auc(y_true, y_score):
        """
        Calculates the AUC-ROC score.

        Parameters
        ----------
        y_true : list or np.ndarray
            True binary labels.
        y_score : list or np.ndarray
            Target scores.

        Returns
        -------
        float
            AUC-ROC score.

        Example
        -------
        >>> y_true = [0, 1, 1, 0]
        >>> y_score = [0.1, 0.9, 0.8, 0.3]
        >>> MetricsCalculator.roc_auc(y_true, y_score)
        0.75
        """
        return roc_auc_score(y_true, y_score)

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """
        Calculates the mean squared error for regression.

        Parameters
        ----------
        y_true : list or np.ndarray
            True values.
        y_pred : list or np.ndarray
            Predicted values.

        Returns
        -------
        float
            Mean squared error.

        Example
        -------
        >>> y_true = [1.0, 2.0, 3.0]
        >>> y_pred = [1.1, 1.9, 3.2]
        >>> MetricsCalculator.mean_squared_error(y_true, y_pred)
        0.013
        """
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        """
        Calculates the mean absolute error for regression.

        Parameters
        ----------
        y_true : list or np.ndarray
            True values.
        y_pred : list or np.ndarray
            Predicted values.

        Returns
        -------
        float
            Mean absolute error.

        Example
        -------
        >>> y_true = [1.0, 2.0, 3.0]
        >>> y_pred = [1.1, 1.9, 3.2]
        >>> MetricsCalculator.mean_absolute_error(y_true, y_pred)
        0.1
        """
        return mean_absolute_error(y_true, y_pred)

    def add_custom_metric(self, name, func):
        """
        Adds a custom metric to the calculator.

        Parameters
        ----------
        name : str
            Name of the custom metric.
        func : callable
            Custom metric function taking y_true and y_pred as arguments.

        Example
        -------
        >>> def custom_metric(y_true, y_pred):
        >>>     return np.mean(y_true == y_pred) * 100
        >>> calc = MetricsCalculator()
        >>> calc.add_custom_metric("custom_acc", custom_metric)
        """
        self.custom_metrics[name] = func

    def compute_custom_metric(self, name, y_true, y_pred):
        """
        Computes a custom metric if it exists.

        Parameters
        ----------
        name : str
            Name of the custom metric.
        y_true : list or np.ndarray
            True labels.
        y_pred : list or np.ndarray
            Predicted labels.

        Returns
        -------
        float
            Computed custom metric.
        """
        if name not in self.custom_metrics:
            raise ValueError(f"Custom metric {name} not found.")
        return self.custom_metrics[name](y_true, y_pred)
