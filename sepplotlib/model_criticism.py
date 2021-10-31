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
    df: pd.DataFrame containing the predictions, actuals and labels.
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
    annotsize: Textsize of annotations.
    annotspacing: Spacing in axes coordinates between annotations.
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
        fgcolors=("#0862ca", "#fd1205"),
        bgcolors=("#cddff4", "#fecfdc"),
        title: str = "",
        titlesize: int = 16,
        xlabel: str = "Predicted probability (p)",
        ylabel: str = "Observation (ordered by p)",
        markersize: int = 100,
        markeralpha: float = 1,
        labelsize: int = 18,
        annotsize: int = 14,
        annotspacing: float = 0.05,
        ticksize: int = 16,
        pad: float = 0.1,
        margin: float = 0.2,
        path: Optional[str] = None,
        dpi: Optional[int] = 200,
    ):
        self.df = df
        self.y_true = y_true
        self.y_pred = y_pred
        self.lab = lab
        self.figsize = figsize
        self.framesize = framesize
        self.n_worst = n_worst
        self.fgcolors = ("#0862ca", "#fd1205")
        self.bgcolors = ("#cddff4", "#fecfdc")
        self.title = title
        self.titlesize = titlesize
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.markersize = markersize
        self.markeralpha = markeralpha
        self.labelsize = labelsize
        self.ticksize = ticksize
        self.annotsize = annotsize
        self.annotspacing = annotspacing
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
        self.axs[0].set_title(self.title, size=self.titlesize, pad=20)
        # Thicken frame.
        for axis in ["top", "bottom", "left", "right"]:
            self.axs[0].spines[axis].set_linewidth(self.framesize)
            self.axs[1].spines[axis].set_linewidth(self.framesize)
        return self

    def prepare_data(self):
        """Prepare sorted data arrays."""
        df = self.df[[self.y_true, self.y_pred, self.lab]].sort_values(
            by=self.y_pred
        )
        df["bgcolor"] = np.where(
            df[self.y_true] == 1, self.bgcolors[1], self.bgcolors[0]
        )
        df = df.reset_index(drop=True)
        # Find highlights.
        worst_fp = df.loc[df[self.y_true] == 0][-self.n_worst :].index
        df["worst_fp"] = np.where(df.index.isin(worst_fp), 1, 0)
        worst_fn = df.loc[df[self.y_true] == 1][: self.n_worst].index
        df["worst_fn"] = np.where(df.index.isin(worst_fn), 1, 0)
        # Conditional coloring.
        df["fgcolor"] = df["bgcolor"]  # Except when...
        df.loc[df.worst_fp == 1, "fgcolor"] = self.fgcolors[0]
        df.loc[df.worst_fn == 1, "fgcolor"] = self.fgcolors[1]
        self.df = df
        return self

    def scatter(self):
        """Plot scatter into top ax."""
        self.axs[0].scatter(
            self.df[self.y_pred],
            self.df.index,
            color=self.df.fgcolor,
            alpha=self.markeralpha,
            s=self.markersize,
            zorder=-3,
        )
        return self

    def density(self):
        """Plot density plot into bottom ax."""
        sns.kdeplot(
            ax=self.axs[1],
            data=self.df,
            x=self.y_pred,
            hue=self.y_true,
            fill=True,
            palette=self.fgcolors,
            legend=False,
            clip=(0.0, 1.0),
            lw=1.5,
        )
        return self

    def rug(self):
        """Add rug to figure."""
        self.rax_y = self.axs[0].inset_axes(
            bounds=[0.96, 0, 0.04, 1], zorder=-1
        )
        for idx, row in self.df.iterrows():
            self.rax_y.hlines(
                y=idx, xmin=0, xmax=1, color=row["fgcolor"], alpha=0.5, lw=3
            )
        # Draw highlight over this background rug.
        for idx, row in self.df.query(
            "worst_fp == 1 or worst_fn == 1"
        ).iterrows():
            self.rax_y.hlines(
                y=idx, xmin=0, xmax=1, color=row["fgcolor"], alpha=0.5, lw=3
            )
        self.rax_y.set_xticks([])
        self.rax_y.set_yticks([])
        self.rax_y.margins(0.02)
        self.rax_y.axis("off")
        return self

    def connect_rug(self):
        """Connect scatter to rug."""
        for idx, row in self.df.loc[self.df.worst_fp == 1].iterrows():
            self.axs[0].hlines(
                y=idx,
                xmin=row[self.y_pred],
                xmax=1 + self.pad,
                color=row["bgcolor"],
                zorder=-2,
                lw=1.5,
            )
        for idx, row in self.df.loc[self.df.worst_fn == 1].iterrows():
            self.axs[0].hlines(
                y=idx,
                xmin=row[self.y_pred],
                xmax=1 + self.pad,
                color=row["bgcolor"],
                zorder=-2,
                lw=1.5,
            )
        return self

    def annotate(self):
        """Annotate highlights."""
        step = 0
        top = 1 - self.axs[0]._ymargin
        trans = self.axs[0].get_xaxis_transform()
        va, ha = ("center", "left")
        # Sort df descending since annotations hang.
        self.df = self.df.sort_values(by=self.y_pred, ascending=False)
        # Annotate the negatives first.
        for idx, row in self.df.loc[self.df.worst_fp == 1].iterrows():
            self.axs[0].annotate(
                row[self.lab],
                xy=(1 + self.pad, idx),
                xycoords="data",
                xytext=(1 + self.pad + self.margin, top - step),
                textcoords=trans,
                va=va,
                ha=ha,
                color=row["fgcolor"],
                size=self.annotsize,
            )
            # Little trick here to actually attach to the left center point.
            self.axs[0].annotate(
                "",
                xy=(1 + self.pad, idx),
                xycoords="data",
                xytext=(0.99 + self.pad + self.margin, top - step),
                textcoords=trans,
                arrowprops=dict(
                    arrowstyle="-",
                    edgecolor=row["fgcolor"],
                    shrinkB=0,
                    shrinkA=0,
                    lw=1.5,
                ),
            )
            step += self.annotspacing
        # Continue with positives.
        for idx, row in self.df.loc[self.df.worst_fn == 1].iterrows():
            self.axs[0].annotate(
                row[self.lab],
                xy=(1 + self.pad, idx),
                xycoords="data",
                xytext=(1 + self.pad + self.margin, top - step),
                textcoords=trans,
                va=va,
                ha=ha,
                color=row["fgcolor"],
                size=self.annotsize,
            )
            self.axs[0].annotate(
                "",
                xy=(1 + self.pad, idx),
                xycoords="data",
                xytext=(0.99 + self.pad + self.margin, top - step),
                textcoords=trans,
                arrowprops=dict(
                    arrowstyle="-",
                    edgecolor=row["fgcolor"],
                    shrinkB=0,
                    shrinkA=0,
                    lw=1.5,
                ),
            )
            step += self.annotspacing

        return self

    def save(self, path, dpi=200):
        """Save figure to path."""
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(self.fig)
