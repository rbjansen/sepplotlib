"""Model criticism plot."""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class ModelCriticismPlot:
    """Produce an model criticism plot.

    Attributes
    ----------
    df: pd.DataFrame containing the predictions of x and y, as well as actuals
        and labels.
    y_true: Column name for the actual observations.
    y_pred: Column name for the model predictions.
    lab: Column name for the row labels.
    figsize: Tuple of (width, height) to pass to figsize.
    framesize: Linewidth of figure frame.
    n_worst: Number of worst error annotations to plot.
    colors: Tuple of colors strings for the negative and positive. For example:
        ("#0000FF", "red").
    title: String title to give to figure. Title is empty by default.
    titlesize: Textsize of figure title.
    xlabel: String label for the x-axis.
    ylabel: String label for the y-axis.
    markersize: Size of markers in scatter plot.
    markeralpha: Alpha of markers in scatter plot.
    labelsize: Textsize of xlabel and ylabel.
    annot_size: Textsize of annotations.
    annot_spacing: Spacing in axes coordinates between annotations.
    ticksize: Size of axes ticks.
    pad: Padding to add inside figure frame, in data coordinates.
    margin: Margin to add outside figure frame, in axes coordinates.
    path: Optional output path to save figure to.
    dpi: Optional integer value for dots per inch. Increase for higher output
        quality.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        y_true: str,
        y_pred: str,
        lab: str,
        figsize: Tuple[int, int] = (5, 7),
        framesize: int = 2,
        n_worst: int = 10,
        title: str = "",
        titlesize: int = 18,
        xlabel: str = "Predicted probability (p)",
        ylabel: str = "Observation (ordered by p)",
        markersize: int = 50,
        markeralpha: float = 1,
        labelsize: int = 16,
        annot_size: int = 12,
        annot_spacing: float = 0.05,
        ticksize: int = 12,
        pad: float = 0.1,
        margin: float = 0.25,
        path: Optional[str] = None,
        dpi: Optional[int] = None,
    ):
        self.y_true = df[y_true].to_numpy()
        self.y_pred = df[y_pred].to_numpy()
        self.lab = df[lab]
        self.figsize = figsize
        self.framesize = framesize
        self.n_worst = n_worst
        self.colors = ("#0862ca", "#fd1205")
        self.linecolors = ("#cddff4", "#fecfdc")
        self.title = title
        self.titlesize = titlesize
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.markersize = markersize
        self.markeralpha = markeralpha
        self.labelsize = labelsize
        self.ticksize = ticksize
        self.annot_size = annot_size
        self.annot_spacing = annot_spacing
        self.pad = pad
        self.margin = margin
        self.path = path
        self.dpi = dpi

        # Make the plot.
        self.fig, self.axs = plt.subplots(
            nrows=2,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [4, 1]},
        )
        (
            ModelCriticismPlot.setup_axs(self)
            .find_highlights()
            .prepare_data()
            .scatter()
            .density()
            .rug()
            .connect_rug()
            .annotate()
        )
        if path is not None:
            ModelCriticismPlot.save(self, path, dpi)

    def __str__(self):
        return f"Model criticism plot of {self.title}."

    def __repr__(self):
        return f"Model criticism plot of {self.title}."

    def setup_axs(self):
        """Set up axs to have common look."""
        for ax in self.axs:
            ax.margins(0.02)
            ax.grid(
                which="major",
                axis="x",
                lw=1,
                color="black",
                alpha=0.1,
            )
            ax.tick_params(labelsize=self.ticksize)
            ax.set_xlim(0 - self.pad, 1 + self.pad)
        self.axs[0].set_ylabel(self.ylabel, size=self.labelsize)
        self.axs[1].set_xlabel(self.xlabel, size=self.labelsize)
        self.axs[1].set_yticks([])
        self.axs[1].set_xticks([0, 0.25, 0.5, 0.75, 1])
        plt.subplots_adjust(hspace=0)
        self.axs[0].set_title(self.title, size=self.titlesize)
        # Thicken frame.
        for axis in ["top", "bottom", "left", "right"]:
            self.axs[0].spines[axis].set_linewidth(self.framesize)
            self.axs[1].spines[axis].set_linewidth(self.framesize)
        return self

    def find_highlights(self):
        """Find predictions to highlight."""
        self.worst_fn = np.where(self.y_true == 1)[0][: self.n_worst]
        self.worst_fp = np.where(self.y_true == 0)[0][-self.n_worst :]
        self.worst_fn = -np.sort(-self.worst_fn)
        self.worst_fp = -np.sort(-self.worst_fp)
        return self

    def prepare_data(self):
        """Prepare sorted data arrays."""
        self.y_pred = np.sort(self.y_pred)
        color_set = np.array(self.linecolors)
        self.sorted_index = np.argsort(self.y_pred)  # Used to index actuals.
        self.y_true = self.y_true[self.sorted_index].astype(int)
        self.lab = self.lab[self.sorted_index]
        self.color_array = color_set[self.y_true]
        self.color_array[self.worst_fp] = self.colors[0]
        self.color_array[self.worst_fn] = self.colors[1]
        return self

    def scatter(self):
        """Plot scatter into top ax."""
        self.axs[0].scatter(
            self.y_pred,
            self.sorted_index,
            color=self.color_array,
            alpha=self.markeralpha,
            s=self.markersize,
        )
        return self

    def density(self):
        """Plot density plot into bottom ax."""
        kde_df = pd.DataFrame({"y_true": self.y_true, "y_pred": self.y_pred})
        sns.kdeplot(
            ax=self.axs[1],
            data=kde_df,
            x="y_pred",
            hue="y_true",
            fill=True,
            palette=self.colors,
            legend=False,
            clip=(0.0, 1.0),
        )
        return self

    def rug(self):
        """Add rug to figure."""
        self.rax_y = self.axs[0].inset_axes(
            bounds=[0.96, 0, 0.045, 1], zorder=4
        )
        for color, value in zip(self.color_array, self.sorted_index):
            self.rax_y.hlines(
                y=value, xmin=0, xmax=1, color=color, alpha=0.5, lw=3
            )
        self.rax_y.set_xticks([])
        self.rax_y.set_yticks([])
        self.rax_y.margins(0.02)
        self.rax_y.axis("off")
        return self

    def connect_rug(self):
        """Connect scatter to rug."""
        linecolor_array = np.array(self.linecolors)[self.y_true]
        for idx in self.worst_fp:
            self.axs[0].hlines(
                y=idx,
                xmin=self.y_pred[idx],
                xmax=1 + self.pad,
                color=linecolor_array[idx],
                zorder=3,
                lw=2,
            )
        for idx in self.worst_fn:
            self.axs[0].hlines(
                y=idx,
                xmin=self.y_pred[idx],
                xmax=1 + self.pad,
                color=linecolor_array[idx],
                zorder=3,
                lw=2,
            )
        return self

    def annotate(self):
        """Annotate highlights."""
        step = 0
        top = 1 - self.axs[0]._ymargin
        trans = self.axs[0].get_xaxis_transform()
        va, ha = ("center", "left")
        # Annotate the negatives first.
        for idx in self.worst_fp:
            self.axs[0].annotate(
                self.lab[idx],
                xy=(1 + self.pad, self.sorted_index[idx]),
                xycoords="data",
                xytext=(1 + self.pad + self.margin, top - step),
                textcoords=trans,
                va=va,
                ha=ha,
                color=self.color_array[idx],
                size=self.annot_size,
            )
            # Little trick here to actually attach to the left center point.
            self.axs[0].annotate(
                "",
                xy=(1 + self.pad, self.sorted_index[idx]),
                xycoords="data",
                xytext=(1 + self.pad + self.margin, top - step),
                textcoords=trans,
                arrowprops=dict(
                    arrowstyle="-",
                    edgecolor=self.color_array[idx],
                    shrinkB=0,
                    shrinkA=0,
                    lw=2,
                ),
            )
            step += self.annot_spacing
        # Continue with positives.
        for idx in self.worst_fn:
            self.axs[0].annotate(
                self.lab[idx],
                xy=(1 + self.pad, self.sorted_index[idx]),
                xycoords="data",
                xytext=(1 + self.pad + self.margin, top - step),
                textcoords=trans,
                va=va,
                ha=ha,
                color=self.color_array[idx],
                size=self.annot_size,
            )
            self.axs[0].annotate(
                "",
                xy=(1 + self.pad, self.sorted_index[idx]),
                xycoords="data",
                xytext=(1 + self.pad + self.margin, top - step),
                textcoords=trans,
                arrowprops=dict(
                    arrowstyle="-",
                    edgecolor=self.color_array[idx],
                    shrinkB=0,
                    shrinkA=0,
                    lw=2,
                ),
            )
            step += self.annot_spacing

        return self

    def save(self, path, dpi=200):
        """Save figure to path."""
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(self.fig)
