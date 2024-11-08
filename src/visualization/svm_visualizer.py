import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from matplotlib.colors import ListedColormap
from plotly.subplots import make_subplots

class SVMVisualizer:
    """
    Enhanced SVMVisualizer provides utilities to visualize SVM decision boundaries,
    support vectors, and margins, with options for using matplotlib, seaborn, or Plotly.

    Parameters
    ----------
    model : object
        Trained SVM model supporting the predict method.
    X : np.ndarray
        2D feature array (n_samples, 2) for visualization.
    y : np.ndarray
        Labels corresponding to X.

    Examples
    --------
    >>> visualizer = SVMVisualizer(model, X, y)
    >>> visualizer.plot_decision_boundary(lib='matplotlib', style='darkgrid')
    >>> visualizer.plot_support_vectors(lib='plotly')
    """

    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y

    def plot_decision_boundary(self, lib="matplotlib", title="SVM Decision Boundary", style="darkgrid", **kwargs):
        """
        Plots the decision boundary, margins, and data points with colors representing classes.

        Parameters
        ----------
        lib : str, optional
            Library to use for plotting. Options: 'matplotlib', 'seaborn', 'plotly'.
        title : str, optional
            Title of the plot.
        style : str, optional
            Style/theme for the plot. Options vary by library.

        Examples
        --------
        >>> visualizer.plot_decision_boundary(lib="seaborn", title="Decision Boundary with RBF Kernel", style="white")
        >>> visualizer.plot_decision_boundary(lib="plotly")
        """
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                            np.arange(y_min, y_max, 0.01))
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        if lib == "matplotlib":
            plt.style.use(style)
            plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(("lightblue", "lightcoral")))
            plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap=ListedColormap(("blue", "red")), edgecolor="k")
            plt.title(title)
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.show()

        elif lib == "seaborn":
            sns.set(style=style)
            plt.figure(figsize=(8, 6))
            plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(("lightblue", "lightcoral")))
            sns.scatterplot(x=self.X[:, 0], y=self.X[:, 1], hue=self.y, palette=["blue", "red"], edgecolor="k")
            plt.title(title)
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.show()

        elif lib == "plotly":
            fig = go.Figure(data=go.Contour(
                z=Z, x=np.arange(x_min, x_max, 0.01), y=np.arange(y_min, y_max, 0.01),
                contours_coloring="fill", colorscale="RdBu", opacity=0.5
            ))
            fig.add_trace(go.Scatter(
                x=self.X[:, 0], y=self.X[:, 1], mode="markers",
                marker=dict(color=self.y, colorscale=["blue", "red"], line=dict(width=1)),
                name="Data Points"
            ))
            fig.update_layout(title=title, xaxis_title="Feature 1", yaxis_title="Feature 2")
            fig.show()

    def plot_support_vectors(self, lib="matplotlib", **kwargs):
        """
        Plots support vectors for the SVM model, if available, with the specified library.

        Parameters
        ----------
        lib : str, optional
            Library to use for plotting. Options: 'matplotlib', 'plotly'.

        Examples
        --------
        >>> visualizer.plot_support_vectors(lib="matplotlib")
        >>> visualizer.plot_support_vectors(lib="plotly")
        """
        if hasattr(self.model, "support_vectors_"):
            if lib == "matplotlib":
                plt.scatter(self.model.support_vectors_[:, 0], self.model.support_vectors_[:, 1],
                            s=100, facecolors="none", edgecolors="k", label="Support Vectors")
                plt.legend()
                plt.title("Support Vectors")
                plt.show()

            elif lib == "plotly":
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=self.model.support_vectors_[:, 0], y=self.model.support_vectors_[:, 1],
                    mode="markers", marker=dict(size=12, symbol="circle-open", color="black"),
                    name="Support Vectors"
                ))
                fig.update_layout(title="Support Vectors")
                fig.show()

        else:
            print("Support vectors not available in the model.")
