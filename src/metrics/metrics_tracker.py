import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import defaultdict
from typing import Callable, Optional, Union
import json
import yaml
import csv


class MetricsTracker:
    """
    Tracks, alerts, and visualizes real-time evaluation metrics for models.

    This class supports live metric tracking with optional smoothing, custom alerting
    conditions, callbacks, and animated visualization. Metrics can be saved and loaded
    in various formats (JSON, YAML, CSV).

    Parameters
    ----------
    smoothing : bool, optional
        Whether to apply exponential smoothing to the metrics. Default is False.
    smooth_factor : float, optional
        Smoothing factor to apply if smoothing is enabled, between 0 and 1.
        Higher values give more weight to recent values. Default is 0.9.

    Attributes
    ----------
    metrics : defaultdict
        Dictionary that stores metric lists by metric name.
    alerts : dict
        Dictionary of alert thresholds and conditions for each metric.
    callbacks : list
        List of callback functions to call after each metric update.
    fig : plt.Figure
        Matplotlib figure used for animated visualization.
    ax : plt.Axes
        Matplotlib axes used for animated visualization.
    ani : FuncAnimation or None
        Animation object for real-time metric plotting.
    animation_running : bool
        Indicator of whether the animation is running.
    """

    def __init__(self, smoothing: bool = False, smooth_factor: float = 0.9):
        self.smoothing = smoothing
        self.smooth_factor = smooth_factor
        self.metrics = defaultdict(list)
        self.alerts = {}
        self.callbacks = []
        self.fig, self.ax = plt.subplots()
        self.ani = None
        self.animation_running = False

    def add_metric(self, name: str, values: Union[float, list]):
        """
        Adds a metric value or list of values to the tracker for a specific metric name.

        Parameters
        ----------
        name : str
            The name of the metric to add.
        values : float or list of float
            Single value or list of values to add to the metric.
        """
        if isinstance(values, list):
            for value in values:
                self._add_single_metric(name, value)
        else:
            self._add_single_metric(name, values)

    def _add_single_metric(self, name: str, value: float):
        """
        Adds a single metric value, applying smoothing if enabled, and triggers alerts and callbacks.

        Parameters
        ----------
        name : str
            The name of the metric.
        value : float
            The value of the metric.
        """
        if self.smoothing and self.metrics[name]:
            value = (
                self.smooth_factor * self.metrics[name][-1]
                + (1 - self.smooth_factor) * value
            )
        self.metrics[name].append(value)
        self._check_alert(name, value)
        self._trigger_callbacks()

    def _check_alert(self, metric: str, value: float):
        """
        Checks if a metric value triggers an alert based on set thresholds.

        Parameters
        ----------
        metric : str
            The metric name to check.
        value : float
            The metric value to check against the alert condition.
        """
        if metric in self.alerts:
            threshold, condition = self.alerts[metric]
            if (condition == "above" and value > threshold) or (
                condition == "below" and value < threshold
            ):
                print(
                    f"Alert: {metric} reached {value}, which is {condition} {threshold}"
                )

    def _trigger_callbacks(self):
        """
        Executes all registered callback functions.
        """
        for callback in self.callbacks:
            callback(self)

    def set_alert(self, metric: str, threshold: float, condition: str = "above"):
        """
        Sets an alert condition for a specific metric.

        Parameters
        ----------
        metric : str
            The name of the metric.
        threshold : float
            The threshold value for the alert.
        condition : str, optional
            Alert condition, either 'above' or 'below'. Default is 'above'.
        """
        self.alerts[metric] = (threshold, condition)

    def register_callback(self, callback: Callable):
        """
        Registers a callback function to be triggered after each metric update.

        Parameters
        ----------
        callback : Callable
            A function to execute after each metric update.
        """
        self.callbacks.append(callback)

    def start_animation(
        self, interval: int = 1000, metrics_to_plot: Optional[list] = None
    ):
        """
        Starts real-time animated plotting of the metrics.

        Parameters
        ----------
        interval : int, optional
            Time interval between plot updates in milliseconds. Default is 1000.
        metrics_to_plot : list of str, optional
            List of metric names to plot. If None, all metrics are plotted.
        """
        if not metrics_to_plot:
            metrics_to_plot = list(self.metrics.keys())
        self.animation_running = True

        def update(frame):
            self.ax.clear()
            for metric in metrics_to_plot:
                data = self.metrics[metric]
                self.ax.plot(data, label=metric)
            self.ax.legend(loc="upper right")
            self.ax.set_xlabel("Batch/Epoch")
            self.ax.set_ylabel("Metric Value")
            self.ax.set_title("Real-Time Metrics Tracking")
            plt.tight_layout()

        self.ani = FuncAnimation(self.fig, update, interval=interval)
        plt.show()

    def stop_animation(self):
        """
        Stops the real-time animated plotting.
        """
        if self.ani:
            self.ani.event_source.stop()
            self.animation_running = False
            print("Animation stopped.")

    def export_metrics(self, file_path: str, format: str = "json"):
        """
        Exports the tracked metrics to a file in JSON, YAML, or CSV format.

        Parameters
        ----------
        file_path : str
            Path to save the exported file.
        format : str, optional
            File format for saving, either 'json', 'yaml', or 'csv'. Default is 'json'.
        """
        if format == "json":
            with open(file_path, "w") as f:
                json.dump(self.metrics, f)
        elif format == "yaml":
            with open(file_path, "w") as f:
                yaml.dump(self.metrics, f)
        elif format == "csv":
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                for key, values in self.metrics.items():
                    writer.writerow([key] + values)

    def load_metrics(self, file_path: str, format: str = "json"):
        """
        Loads metrics from a file in JSON, YAML, or CSV format.

        Parameters
        ----------
        file_path : str
            Path to the file to load.
        format : str, optional
            File format to load, either 'json', 'yaml', or 'csv'. Default is 'json'.
        """
        if format == "json":
            with open(file_path, "r") as f:
                self.metrics.update(json.load(f))
        elif format == "yaml":
            with open(file_path, "r") as f:
                self.metrics.update(yaml.safe_load(f))
        elif format == "csv":
            with open(file_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    self.metrics[row[0]].extend(map(float, row[1:]))

    def reset(self):
        """
        Resets all stored metrics and alert conditions.
        """
        self.metrics.clear()
        self.alerts.clear()

    def get_metrics(self):
        """
        Returns all tracked metrics as a dictionary.

        Returns
        -------
        dict
            Dictionary containing all tracked metrics.
        """
        return dict(self.metrics)
