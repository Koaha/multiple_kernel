import numpy as np
from sklearn.model_selection import (
    cross_val_score,
    KFold,
    ParameterGrid,
    ParameterSampler,
)
from concurrent.futures import ProcessPoolExecutor
import joblib
import optuna
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import yaml
import toml
import json


class HyperparameterTuner:
    """
    Hyperparameter tuner for various SVM solvers, integrating Optuna, grid search, random search,
    cross-validation, parallelization, warm start, early stopping, and feature selection.

    Parameters
    ----------
    model_class : class
        The SVM solver class (e.g., LaSVM, coreSVM, NySVM).
    param_grid : dict
        Grid of parameters for tuning.
    n_trials : int, default=100
        Number of trials for Optuna or random search optimization.
    n_jobs : int, default=-1
        Number of parallel jobs. Use -1 for all available cores.
    cv_folds : int, default=5
        Number of cross-validation folds.
    scoring : str, default='accuracy'
        Scoring metric for evaluation.
    enable_feature_selection : bool, default=True
        Whether to perform feature selection.
    early_stopping_rounds : int, default=5
        Number of rounds without improvement to trigger early stopping.
    method : str, default='optuna'
        Search method for tuning ('optuna', 'grid', or 'random').

    Attributes
    ----------
    study : optuna.Study
        Optuna study object for tuning results if Optuna is used.
    best_params_ : dict
        Best parameters found during tuning.
    best_score_ : float
        Best cross-validation score achieved.
    feature_importances_ : np.ndarray
        Feature importances if feature selection is enabled.
    """

    def __init__(
        self,
        model_class,
        param_grid,
        n_trials=100,
        n_jobs=-1,
        cv_folds=5,
        scoring="accuracy",
        enable_feature_selection=True,
        early_stopping_rounds=5,
        method="optuna",
    ):
        self.model_class = model_class
        self.param_grid = param_grid
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.enable_feature_selection = enable_feature_selection
        self.early_stopping_rounds = early_stopping_rounds
        self.method = method
        self.study = None
        self.best_params_ = None
        self.best_score_ = None
        self.feature_importances_ = None

    def objective(self, trial, X, y):
        """
        Objective function for Optuna optimization.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object.
        X : np.ndarray
            Feature data.
        y : np.ndarray
            Labels.

        Returns
        -------
        float
            Cross-validated score.
        """
        params = {
            key: (
                trial.suggest_categorical(key, values)
                if isinstance(values, list)
                else trial.suggest_float(key, values[0], values[1])
            )
            for key, values in self.param_grid.items()
        }

        model = self.model_class(**params)

        if self.enable_feature_selection:
            selector = RFE(RandomForestClassifier(), n_features_to_select=10, step=1)
            X = selector.fit_transform(X, y)
            self.feature_importances_ = selector.get_support()

        kf = KFold(n_splits=self.cv_folds)
        scores = cross_val_score(
            model, X, y, cv=kf, scoring=self.scoring, n_jobs=self.n_jobs
        )
        return np.mean(scores)

    def tune(self, X, y):
        """
        Run the hyperparameter tuning.

        Parameters
        ----------
        X : np.ndarray
            Feature data.
        y : np.ndarray
            Labels.
        """
        if self.method == "optuna":
            # Optuna tuning with early stopping
            self.study = optuna.create_study(
                direction="maximize",
                pruner=optuna.pruners.MedianPruner(self.early_stopping_rounds),
            )
            self.study.optimize(
                lambda trial: self.objective(trial, X, y),
                n_trials=self.n_trials,
                n_jobs=self.n_jobs,
            )
            self.best_params_ = self.study.best_params
            self.best_score_ = self.study.best_value

        elif self.method == "grid":
            # Grid search over all parameter combinations
            best_score = -np.inf
            for params in ParameterGrid(self.param_grid):
                score = self._evaluate_model(params, X, y)
                if score > best_score:
                    best_score = score
                    self.best_params_ = params
            self.best_score_ = best_score

        elif self.method == "random":
            # Random search over random samples of parameter space
            best_score = -np.inf
            for params in ParameterSampler(
                self.param_grid, n_iter=self.n_trials, random_state=0
            ):
                score = self._evaluate_model(params, X, y)
                if score > best_score:
                    best_score = score
                    self.best_params_ = params
            self.best_score_ = best_score

    def _evaluate_model(self, params, X, y):
        """
        Evaluate model with specified parameters.

        Parameters
        ----------
        params : dict
            Parameters to evaluate.
        X : np.ndarray
            Feature data.
        y : np.ndarray
            Labels.

        Returns
        -------
        float
            Cross-validated score.
        """
        model = self.model_class(**params)

        if self.enable_feature_selection:
            selector = RFE(RandomForestClassifier(), n_features_to_select=10, step=1)
            X = selector.fit_transform(X, y)
            self.feature_importances_ = selector.get_support()

        kf = KFold(n_splits=self.cv_folds)
        scores = cross_val_score(
            model, X, y, cv=kf, scoring=self.scoring, n_jobs=self.n_jobs
        )
        return np.mean(scores)

    def get_best_model(self, X, y):
        """
        Train the best model on the entire dataset using the best hyperparameters.

        Parameters
        ----------
        X : np.ndarray
            Feature data.
        y : np.ndarray
            Labels.

        Returns
        -------
        model : fitted model
            The trained model with the best parameters.
        """
        model = self.model_class(**self.best_params_)
        model.fit(X, y)
        return model

    def save_results(
        self,
        filename="tuning_results",
        file_format="yaml",
        summary_filename="optuna_summary.txt",
    ):
        """
        Save the tuning results to the specified format (YAML, TOML, JSON) for parameters and score,
        and Optuna summary to a text file if Optuna was used.

        Parameters
        ----------
        filename : str, default='tuning_results'
            Base filename to save parameter results (file extension added automatically).
        file_format : str, default='yaml'
            Format to save the file in. Options are 'yaml', 'toml', or 'json'.
        summary_filename : str, default='optuna_summary.txt'
            Filename to save Optuna tuning summary if Optuna was used.
        """
        # Determine the appropriate extension and dump function based on the format
        if file_format == "yaml":
            file_path = f"{filename}.yaml"
            with open(file_path, "w") as f:
                yaml.dump(
                    {"best_params": self.best_params_, "best_score": self.best_score_},
                    f,
                )

        elif file_format == "toml":
            file_path = f"{filename}.toml"
            with open(file_path, "w") as f:
                toml.dump(
                    {"best_params": self.best_params_, "best_score": self.best_score_},
                    f,
                )

        elif file_format == "json":
            file_path = f"{filename}.json"
            with open(file_path, "w") as f:
                json.dump(
                    {"best_params": self.best_params_, "best_score": self.best_score_},
                    f,
                )

        else:
            raise ValueError(
                "Invalid file format. Choose from 'yaml', 'toml', or 'json'."
            )

        # Save Optuna summary as plain text
        if self.method == "optuna":
            with open(summary_filename, "w") as f:
                f.write(f"Best Score: {self.best_score_}\n")
                f.write("Best Parameters:\n")
                f.write(
                    yaml.dump(self.best_params_)
                    if file_format == "yaml"
                    else (
                        toml.dumps(self.best_params_)
                        if file_format == "toml"
                        else json.dumps(self.best_params_, indent=2)
                    )
                )

    def load_results(self, filename="tuning_results", file_format="yaml"):
        """
        Load tuning results from the specified format (YAML, TOML, JSON) for parameter values and score.

        Parameters
        ----------
        filename : str, default='tuning_results'
            Base filename to load tuning results from (file extension added automatically).
        file_format : str, default='yaml'
            Format to load the file in. Options are 'yaml', 'toml', or 'json'.
        """
        if file_format == "yaml":
            file_path = f"{filename}.yaml"
            with open(file_path, "r") as f:
                results = yaml.safe_load(f)

        elif file_format == "toml":
            file_path = f"{filename}.toml"
            with open(file_path, "r") as f:
                results = toml.load(f)

        elif file_format == "json":
            file_path = f"{filename}.json"
            with open(file_path, "r") as f:
                results = json.load(f)

        else:
            raise ValueError(
                "Invalid file format. Choose from 'yaml', 'toml', or 'json'."
            )

        # Set attributes from loaded data
        self.best_params_ = results["best_params"]
        self.best_score_ = results["best_score"]
