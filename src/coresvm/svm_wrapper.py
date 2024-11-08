# svm_wrapper.py

from solvers import CoreSVM, NySVM, LaSVM
from src.kernels.nystrom import NystromApproximation
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib
import h5py
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


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

    def __init__(
        self,
        solver_type="coresvm",
        config=None,
        use_nystrom=False,
        nystrom_params=None,
        **kwargs,
    ):
        # Check if Nyström approximation is needed
        if use_nystrom:
            if nystrom_params is None:
                raise ValueError(
                    "Nyström parameters must be provided when use_nystrom is True."
                )
            nystrom = NystromApproximation(**nystrom_params)
            self.kernel_matrix = nystrom.compute_nystrom_approximation()
            kwargs["kernel"] = self.kernel_matrix
        else:
            self.kernel_matrix = None

        # Initialize the solver
        if solver_type == "coresvm":
            self.model = CoreSVM(**kwargs)
        elif solver_type == "nysvm":
            self.model = NySVM(**kwargs)
        elif solver_type == "lasvm":
            self.model = LaSVM(**kwargs)
        else:
            raise ValueError(
                "Invalid solver type. Choose from 'coresvm', 'nysvm', 'lasvm'."
            )
        self.config = config or {}

    @staticmethod
    def _initialize_model_from_params(params):
        """
        Initialize an SVM model instance using parameters from HDF5.

        Parameters
        ----------
        params : dict
            Dictionary of model parameters.

        Returns
        -------
        model : object
            Instantiated model with provided parameters.
        """
        # Example assumes a generic SVM class with `set_params` method.
        from src.solvers.coresvm import (
            CoreSVM,
        )  # Update this with actual model class if needed.

        model = CoreSVM()  # Replace with specific model initialization if needed.
        model.set_params(**params)
        return model

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

    def save_model(self, file_path, format="onnx", compress=False):
        """
        Save the model to a file in the specified format.

        Parameters
        ----------
        file_path : str
            Path where the model should be saved.
        format : str, optional
            Format in which to save the model. Options are 'onnx', 'joblib', or 'hdf5'.
        compress : bool, optional
            If True, compress the saved file (joblib and HDF5 formats only).

        Example
        -------
        Save model in different formats:

        ```python
        # Save in ONNX format
        wrapper = SVMWrapper(trained_model, config={"description": "Trained SVM model"})
        wrapper.save_model("model.onnx", format="onnx")

        # Save in Joblib format with compression
        wrapper.save_model("model.joblib", format="joblib", compress=True)

        # Save in HDF5 format with compression
        wrapper.save_model("model.h5", format="hdf5", compress=True)
        ```
        """
        logging.info(f"Saving model in {format} format to {file_path}.")

        if format == "onnx":
            initial_type = [
                ("float_input", FloatTensorType([None, self.model.X.shape[1]]))
            ]
            onnx_model = convert_sklearn(self.model, initial_types=initial_type)
            with open(file_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            logging.info("Model saved in ONNX format.")

        elif format == "joblib":
            joblib.dump((self.model, self.config), file_path, compress=compress)
            logging.info(
                f"Model saved in Joblib format{' with compression' if compress else ''}."
            )

        elif format == "hdf5":
            with h5py.File(file_path, "w") as f:
                # Save model parameters as datasets
                for key, value in self.model.get_params().items():
                    f.attrs[key] = value
                # Save metadata if available
                for key, value in self.config.items():
                    f.attrs[f"config_{key}"] = value
                logging.info(
                    f"Model and metadata saved in HDF5 format{' with compression' if compress else ''}."
                )

        else:
            raise ValueError(
                "Unsupported format. Choose from 'onnx', 'joblib', or 'hdf5'."
            )

    @classmethod
    def load_model(cls, file_path, format=None):
        """
        Load a model from a file, auto-detecting the format if not specified.

        Parameters
        ----------
        file_path : str
            Path to the file from which the model is to be loaded.
        format : str, optional
            Format of the saved model. Options are 'onnx', 'joblib', or 'hdf5'.
            If None, auto-detects based on file extension.

        Returns
        -------
        SVMWrapper
            An instance of SVMWrapper with the loaded model and configuration.

        Example
        -------
        Load model and make predictions:

        ```python
        # Load model (auto-detect format based on file extension)
        loaded_wrapper = SVMWrapper.load_model("model.onnx")
        predictions = loaded_wrapper.predict(X_test)

        # Load model explicitly specifying format
        loaded_wrapper = SVMWrapper.load_model("model.joblib", format="joblib")
        predictions = loaded_wrapper.predict(X_test)
        ```
        """
        logging.info(f"Loading model from {file_path}.")
        if format is None:
            if file_path.endswith(".onnx"):
                format = "onnx"
            elif file_path.endswith(".joblib"):
                format = "joblib"
            elif file_path.endswith(".h5") or file_path.endswith(".hdf5"):
                format = "hdf5"
            else:
                raise ValueError(
                    "Cannot determine format from file extension. Please specify format explicitly."
                )

        if format == "onnx":
            onnx_model = onnx.load(file_path)
            onnx.checker.check_model(onnx_model)
            logging.info("ONNX model loaded successfully.")
            return cls(onnx_model)

        elif format == "joblib":
            model, config = joblib.load(file_path)
            logging.info("Joblib model loaded successfully.")
            return cls(model, config)

        elif format == "hdf5":
            with h5py.File(file_path, "r") as f:
                config = {
                    key[7:]: f.attrs[key]
                    for key in f.attrs
                    if key.startswith("config_")
                }
                model_params = {
                    key: f.attrs[key]
                    for key in f.attrs
                    if not key.startswith("config_")
                }
            model = cls._initialize_model_from_params(model_params)
            logging.info("HDF5 model loaded successfully.")
            return cls(model, config)

        else:
            raise ValueError(
                "Unsupported format. Choose from 'onnx', 'joblib', or 'hdf5'."
            )

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
        if isinstance(self.model, onnx.ModelProto):
            ort_session = ort.InferenceSession(self.model.SerializeToString())
            input_name = ort_session.get_inputs()[0].name
            predictions = ort_session.run(None, {input_name: X.astype(np.float32)})
            return predictions[0]
        else:
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
            raise NotImplementedError(
                f"{self.model.__class__.__name__} does not support parameter setting."
            )

    def cross_validate(
        self,
        X,
        Y,
        k_folds=5,
        metrics=["accuracy", "precision", "recall", "f1", "roc_auc"],
    ):
        """
        Perform cross-validation with specified metrics.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features).
        Y : np.ndarray
            Labels, shape (n_samples,).
        k_folds : int, optional, default=5
            Number of folds for cross-validation.
        metrics : list of str, optional, default=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            Metrics to compute for each fold.

        Returns
        -------
        dict
            Average values of each metric across folds.
        """
        cross_validator = CrossValidation(self, X, Y, k_folds=k_folds, metrics=metrics)
        return cross_validator.evaluate_model()
