import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class RealTimeVisualizer:
    """
    Real-time visualization utility for tracking model metrics.
    Supports dynamic metric updates and animations.

    Parameters
    ----------
    metrics_data : dict
        Dictionary of metrics where keys are metric names and values are lists tracking metric values over time.
    refresh_rate : int, optional
        Frequency of plot updates in milliseconds. Default is 1000.

    Examples
    --------
    >>> metrics = {"accuracy": [], "loss": []}
    >>> visualizer = RealTimeVisualizer(metrics)
    >>> visualizer.animate()
    """

    def __init__(self, metrics_data, refresh_rate=1000):
        self.metrics_data = metrics_data
        self.refresh_rate = refresh_rate
        self.fig, self.ax = plt.subplots(len(metrics_data), 1, figsize=(8, 6))

    def update(self, metric_name, new_value):
        """
        Adds a new value to the specified metric.

        Parameters
        ----------
        metric_name : str
            Name of the metric.
        new_value : float
            New value to add to the metric.

        Example
        -------
        >>> visualizer.update("accuracy", 0.85)
        """
        if metric_name in self.metrics_data:
            self.metrics_data[metric_name].append(new_value)

    def animate(self):
        """
        Sets up and displays the animated plot.
        """

        def update(frame):
            for idx, (metric_name, values) in enumerate(self.metrics_data.items()):
                self.ax[idx].clear()
                self.ax[idx].plot(values, label=metric_name)
                self.ax[idx].legend(loc="upper right")
                self.ax[idx].set_title(f"Real-Time {metric_name}")
            plt.tight_layout()

        animation = FuncAnimation(self.fig, update, interval=self.refresh_rate)
        plt.show()
