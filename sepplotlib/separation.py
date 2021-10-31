"""Simple one-dimensional separation plot."""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SeparationPlot:
    """Make a one-dimensional separation plot.

    Attributes
    ----------
    df: pd.DataFrame containing the predictions and actuals.
    y_true: Column name for the actual observations.
    y_pred: Column name for the model predictions.
    title: String title to give to figure. Title is empty by default.
    figsize: Tuple of figsize by width and height in inches.
    colors: Tuple of string colors for the negative and positive. Example:
        ("blue": "#E34A33").
    axis_off: Bool turning axis ticks and ticklabels off.
    path: String representation of path to write figure to. Default is set to
        None: saving is optional.
    dpi: Optional integer value for dots per inch. Increase for higher output
        quality.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        y_true: str,
        y_pred: str,
        title: str = "",
        figsize: Tuple[float, float] = (9, 1.5),
        colors: Tuple[str, str] = ("#FEF0D9", "#E34A33"),
        axis_off: bool = False,
        path: Optional[str] = None,
        dpi: Optional[int] = 200,
    ):
        self.y_true = np.array(df[y_true])
        self.y_pred = np.array(df[y_pred])
        self.title = title
        self.figsize = figsize
        self.colors = colors
        self.axis_off = axis_off
        # Make figure.
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        SeparationPlot.set_frame(self).plot()
        if path is not None:
            SeparationPlot.save(self, path, dpi)

    def __str__(self):
        return f"Separation plot of {self.title}."

    def __repr__(self):
        return f"Separation plot of {self.title}."

    def set_frame(self):
        """Set up a matplotlib fig and ax."""
        self.ax.set_title(self.title, fontsize=15)
        # Remove unnecessary spines.
        for spine in ("top", "right", "bottom", "left"):
            self.ax.spines[spine].set_visible(False)
        # Tighten layout.
        plt.xlim(0, len(self.y_pred))
        plt.tight_layout()
        if self.axis_off:
            self.ax.axis("off")
        return self

    def plot(self):
        """Plot separation plot into fig."""
        # Prepare params.
        if np.isnan(self.y_pred).any() or np.isnan(self.y_true).any():
            raise RuntimeError("Missing values found in the provided series.")
        color_array = np.array(self.colors)
        sorted_index = np.argsort(self.y_pred)
        sorted_obs = self.y_true[sorted_index].astype(int)
        # Plot the different bars.
        self.ax.bar(
            np.arange(len(self.y_true)),
            np.ones(len(self.y_true)),
            width=1,
            color=color_array[sorted_obs],
            edgecolor="none",
        )
        # Plot p line.
        self.ax.plot(
            np.arange(len(self.y_pred)),
            self.y_pred[sorted_index],
            "k",
            linewidth=1,
        )
        # Create expected value bar.
        self.ax.vlines([(1 - self.y_pred[sorted_index]).sum()], [0], [1])
        return self

    def save(self, path, dpi=200):
        """Save figure to path."""
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(self.fig)
