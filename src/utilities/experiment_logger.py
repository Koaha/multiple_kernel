import logging
import json
import mlflow
import wandb
from datetime import datetime
from typing import Optional, Dict

class ExperimentLogger:
    """
    ExperimentLogger handles logging for parameters, metrics, and kernel choices.
    Supports local logging and integration with MLflow and Weights and Biases (W&B).

    Parameters
    ----------
    experiment_name : str
        Name of the experiment for logging and tracking.
    log_to_file : bool, optional
        If True, logs will be saved to a file.
    use_mlflow : bool, optional
        If True, logs will be saved to an MLflow experiment.
    use_wandb : bool, optional
        If True, logs will be saved to Weights and Biases (W&B).
    wandb_project : str, optional
        W&B project name. Required if `use_wandb` is True.

    Examples
    --------
    >>> logger = ExperimentLogger("SVM Experiment", log_to_file=True, use_mlflow=True, use_wandb=True, wandb_project="SVM_Project")
    >>> logger.log_params({"C": 1.0, "kernel": "RBF"})
    >>> logger.log_metrics({"accuracy": 0.92, "precision": 0.88})
    >>> logger.end_experiment()
    """

    def __init__(self, experiment_name: str, log_to_file: bool = True, use_mlflow: bool = False,
                 use_wandb: bool = False, wandb_project: Optional[str] = None):
        self.experiment_name = experiment_name
        self.log_to_file = log_to_file
        self.use_mlflow = use_mlflow
        self.use_wandb = use_wandb

        # Initialize local logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        if log_to_file:
            log_filename = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_filename)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Initialize MLflow
        if use_mlflow:
            mlflow.set_experiment(experiment_name)
            self.mlflow_run = mlflow.start_run()

        # Initialize W&B
        if use_wandb:
            if not wandb_project:
                raise ValueError("wandb_project must be specified if use_wandb is True")
            wandb.init(project=wandb_project, name=experiment_name)

    def log_params(self, params: Dict[str, any]):
        """
        Logs parameters for the experiment.

        Parameters
        ----------
        params : dict
            Dictionary of parameters to log.
        """
        if self.log_to_file:
            self.logger.info(f"Parameters: {json.dumps(params)}")

        if self.use_mlflow:
            mlflow.log_params(params)

        if self.use_wandb:
            wandb.config.update(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Logs performance metrics for the experiment.

        Parameters
        ----------
        metrics : dict
            Dictionary of metrics to log.
        step : int, optional
            Step or epoch of the metrics being logged.
        """
        if self.log_to_file:
            self.logger.info(f"Metrics: {json.dumps(metrics)}")

        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)

        if self.use_wandb:
            wandb.log(metrics, step=step)

    def log_kernel_choice(self, kernel_name: str, kernel_params: Optional[Dict[str, any]] = None):
        """
        Logs the kernel choice and its parameters.

        Parameters
        ----------
        kernel_name : str
            Name of the kernel function.
        kernel_params : dict, optional
            Parameters for the kernel function.
        """
        kernel_info = {"kernel_name": kernel_name, "kernel_params": kernel_params or {}}
        if self.log_to_file:
            self.logger.info(f"Kernel Choice: {json.dumps(kernel_info)}")

        if self.use_mlflow:
            mlflow.log_param("kernel_name", kernel_name)
            mlflow.log_params(kernel_params or {})

        if self.use_wandb:
            wandb.config.update(kernel_info)

    def end_experiment(self):
        """
        Ends the logging session and closes connections to MLflow or W&B.
        """
        if self.use_mlflow:
            mlflow.end_run()
        if self.use_wandb:
            wandb.finish()
        if self.log_to_file:
            logging.shutdown()
