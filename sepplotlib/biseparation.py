"""Bi-separation plot."""

from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class BiseparationPlot:
    """Make a biseparation plot.

    Attributes
    ----------
    df: pd.DataFrame containing the predictions of x and y, as well as actuals
        and labels.
    x: Column name for the model predictions on the x-axis.
    y: Column name for the model predictions on the y-axis.
    obs: Column name for the actual observations.
    lab: Column name for the row labels.
    framesize: Linewidth of figure frame.
    width: Width of figure frame. Aspect ratio is set equal.
    n_worst: Integer number of most divergent predictions to annotate.
    markersize: Size of scatter plot markers.
    margin: Float margin outside figure frame per axis coords.
    pad: Float padding inside figure frame per data coords.
    colors: Tuple of colors strings for the negative and positive. For example:
        ("#0000FF", "red").
    bg_alpha: Alpha of background scatter.
    fg_alpha: Alpha of foreground scatter.
    con_alpha: Alpha of connections between scatter and rug.
    titlesize: Integer value for title size.
    labelsize: Integer value for label size.
    labelspacing: Float value for space between the annotation labels.
    path: Optional output path to save figure to.
    dpi: Optional integer value for dots per inch. Increase for higher output
        quality.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        obs: str,
        lab: str,
        framesize: float = 3,
        width: int = 10,
        n_worst: int = 5,
        markersize: int = 400,
        margin: float = 0.1,
        pad: float = 0.05,
        colors: Tuple[str, str] = ("#0862ca", "#fd1205"),
        bgcolors: Tuple[str, str] = ("#cddff4", "#fecfdc"),
        bg_alpha: float = 1,
        fg_alpha: float = 0.9,
        con_alpha: float = 0.2,
        titlesize: int = 16,
        labelsize: int = 12,
        labelspacing: float = 0.05,
        path: Optional[str] = None,
        dpi: Optional[int] = None,
    ):
        self.df = df.copy()
        self.x = x
        self.y = y
        self.obs = obs
        self.lab = lab
        self.framesize = framesize
        self.width = width
        self.n_worst = n_worst
        self.markersize = markersize
        self.margin = margin
        self.pad = pad
        self.colors = colors
        self.bgcolors = bgcolors
        self.bg_alpha = bg_alpha
        self.fg_alpha = fg_alpha
        self.con_alpha = con_alpha
        self.titlesize = titlesize
        self.labelsize = labelsize
        self.labelspacing = labelspacing

        self.fig, self.ax = plt.subplots(
            figsize=(self.width, self.width),
        )
        lims = (0 - self.pad, 1 + self.pad)

        (
            BiseparationPlot.set_frame(self)
            .find_highlights()
            .scatter()
            .rug()
            .connect_rug()
            .annotate(model=x, refloc=lims[0], margin=(0 - margin), axis="x")
            .annotate(model=y, refloc=lims[1], margin=(1 + margin), axis="y")
        )
        if path is not None:
            BiseparationPlot.save(self, path, dpi)

    def __str__(self):
        return f"Biseparation plot of {self.x} compared to {self.y}"

    def __repr__(self):
        return f"Biseparation plot of {self.x} compared to {self.y}"

    def set_frame(self):
        """Set up figure frame."""
        plt.tick_params(
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
        )
        # Thicken frame.
        for axis in ["top", "bottom", "left", "right"]:
            self.ax.spines[axis].set_linewidth(self.framesize)
        # Diagonal.
        self.ax.plot(
            [0, 1],
            [0, 1],
            transform=self.ax.transAxes,
            lw=self.framesize,
            color="black",
            zorder=1,
        )
        # Axis labels.
        self.ax.set_ylabel(self.y, size=self.titlesize, labelpad=20)
        self.ax.set_xlabel(self.x, size=self.titlesize, labelpad=20)
        self.ax.xaxis.set_label_position("top")

        return self

    def find_highlights(self):
        """Find predictions to highlight.

        We take the predictions for each binary outcome, for each model, that
        are the furthest apart from the other model. In effect this selects
        the predictions furthest away from the diagonal.
        """

        # First, get the index of sorted predictions as a column.
        for model in (self.x, self.y):
            self.df[f"order_{model}"] = (
                self.df[model]
                .sort_values()
                .reset_index()
                .sort_values("index")
                .index
            )
        # Then, compute the delta between the two orders.
        self.df["delta_order"] = (
            self.df[f"order_{self.x}"] - self.df[f"order_{self.y}"]
        )
        # Negative values on the delta indicate better negative preds for X,
        # or better positive predictions for Y.
        self.df[f"neg_{self.x}"] = np.where(
            self.df.index.isin(
                (
                    self.df.sort_values("delta_order", ascending=True).query(
                        f"delta_order < 0 & {self.obs} == 0"
                    )[
                        : self.n_worst
                    ]  # Ascending to get negatives on top.
                ).index
            ),
            1,
            0,
        )
        self.df[f"pos_{self.y}"] = np.where(
            self.df.index.isin(
                (
                    self.df.sort_values("delta_order", ascending=True).query(
                        f"delta_order < 0 & {self.obs} == 1"
                    )[: self.n_worst]
                ).index
            ),
            1,
            0,
        )
        # And the inverse for positive values...
        self.df[f"neg_{self.y}"] = np.where(
            self.df.index.isin(
                (
                    self.df.sort_values("delta_order", ascending=False).query(
                        f"delta_order > 0 & {self.obs} == 0"
                    )[: self.n_worst]
                ).index
            ),
            1,
            0,
        )
        self.df[f"pos_{self.x}"] = np.where(
            self.df.index.isin(
                (
                    self.df.sort_values("delta_order", ascending=False).query(
                        f"delta_order > 0 & {self.obs} == 1"
                    )[: self.n_worst]
                ).index
            ),
            1,
            0,
        )
        # Collect highlights by model and across.
        for model in (self.x, self.y):
            self.df[f"highlight_{model}"] = np.where(
                (self.df[f"pos_{model}"] == 1)
                | (self.df[f"neg_{model}"] == 1),
                1,
                0,
            )
        self.df["highlight_all"] = np.where(
            self.df[self.df.columns[-2:]].sum(axis=1) > 0, 1, 0
        )  # Per last two columns set.

        return self

    def scatter(self):
        """Make scatter plot."""
        self.ax.scatter(
            x=self.df.loc[self.df["highlight_all"] != 1, self.x],
            y=self.df.loc[self.df["highlight_all"] != 1, self.y],
            color=self.df.loc[self.df["highlight_all"] != 1, self.obs].apply(
                lambda x: self.bgcolors[0] if x == 0 else self.bgcolors[1]
            ),
            alpha=self.bg_alpha,
            zorder=0,
            linewidths=3,
            s=self.markersize,
        )
        # Separate scatter for highlights (z-order can't be conditional).
        self.ax.scatter(
            x=self.df.loc[self.df["highlight_all"] == 1, self.x],
            y=self.df.loc[self.df["highlight_all"] == 1, self.y],
            color=self.df.loc[self.df["highlight_all"] == 1, self.obs].apply(
                lambda x: self.colors[0] if x == 0 else self.colors[1]
            ),
            alpha=self.fg_alpha,
            zorder=2,
            linewidths=2,
            edgecolor="black",
            s=1.25 * self.markersize,  # A little bigger to create perspective.
        )

        return self

    def rug(self):
        """Add axes with rugs to figure."""
        self.rax_y = self.ax.inset_axes(bounds=[0.97, 0, 0.03, 1], zorder=0)
        for index, value in self.df[self.y].items():
            if self.df.loc[index, "highlight_all"] == 1:
                color_set = self.colors
            else:
                color_set = self.bgcolors
            color = (
                color_set[0]
                if self.df.loc[index, self.obs] == 0
                else color_set[1]
            )
            self.rax_y.hlines(
                y=value, xmin=0, xmax=1, color=color, alpha=0.5, lw=3
            )
        self.rax_y.set_xticks([])
        self.rax_y.set_yticks([])
        self.rax_y.margins(0.02)
        self.rax_y.axis("off")
        # And the x-rug.
        self.rax_x = self.ax.inset_axes(bounds=[0, 0, 1, 0.03], zorder=0)
        for index, value in self.df[self.x].items():
            if self.df.loc[index, "highlight_all"] == 1:
                color_set = self.colors
            else:
                color_set = self.bgcolors
            color = (
                color_set[0]
                if self.df.loc[index, self.obs] == 0
                else color_set[1]
            )
            self.rax_x.vlines(
                x=value, ymin=0, ymax=1, color=color, alpha=0.5, lw=3
            )
        self.rax_x.set_xticks([])
        self.rax_x.set_yticks([])
        self.rax_x.margins(0.02)
        self.rax_x.axis("off")
        # Set some space for the rugs.
        xpad, ypad = (0 - self.pad, 1 + self.pad)
        self.ax.set_xlim(xpad, ypad)
        self.ax.set_ylim(xpad, ypad)
        self.rax_y.set_ylim(xpad, ypad)
        self.rax_x.set_xlim(xpad, ypad)

        return self

    def connect_rug(self):
        """Connect scatter to rugs."""
        for index, value in self.df.loc[
            self.df[f"highlight_{self.y}"] == 1
        ].iterrows():
            color = (
                self.colors[0]
                if self.df.loc[index, self.obs] == 0
                else self.colors[1]
            )
            self.ax.hlines(
                y=value[self.y],
                xmin=value[self.x],
                xmax=1 + self.pad,
                color=color,
                alpha=self.con_alpha,
                zorder=3,
                lw=3,
            )
        # vlines for the x-axis.
        for index, value in self.df.loc[
            self.df[f"highlight_{self.x}"] == 1
        ].iterrows():
            color = (
                self.colors[0]
                if self.df.loc[index, self.obs] == 0
                else self.colors[1]
            )
            self.ax.vlines(
                x=value[self.x],
                ymin=value[self.y],
                ymax=0 - self.pad,
                color=color,
                alpha=self.con_alpha,
                zorder=3,
                lw=3,
            )
        return self

    def annotate(self, model, refloc, margin, axis):
        """Annotate the highlights.

        Parameters
        ----------
        model: Column name of the model to annotate.
        refloc: Point to link annotation up to.
        margin: Margin from figure frame.
        axis: Axis the model is on: "x" or "y".
        """
        step = 0
        start_loc = self.df.loc[self.df[f"neg_{model}"] == 1, model].min()
        if axis == "x":
            trans = self.ax.get_xaxis_transform()
            rotation = -90
            va, ha = ("top", "center")
        else:
            trans = self.ax.get_yaxis_transform()
            rotation = 0
            va, ha = ("center", "left")
        # Annotate the negatives first.
        for _, value in (
            self.df.loc[self.df[f"neg_{model}"] == 1]
            .sort_values(model)
            .iterrows()
        ):
            self.ax.annotate(
                value[self.lab],
                xy=(value[model], refloc)
                if axis == "x"
                else (refloc, value[model]),
                xycoords="data",
                xytext=(start_loc + step, margin)
                if axis == "x"
                else (margin, start_loc + step),
                textcoords=trans,
                va=va,
                ha=ha,
                color=self.colors[0],
                rotation=rotation,
                size=self.labelsize,
            )
            # Little trick here to actually attach to the left center point.
            self.ax.annotate(
                "",
                xy=(value[model], refloc)
                if axis == "x"
                else (refloc, value[model]),
                xycoords="data",
                xytext=(start_loc + step, margin)
                if axis == "x"
                else (margin, start_loc + step),
                textcoords=trans,
                arrowprops=dict(
                    arrowstyle="-",
                    edgecolor=self.colors[0],
                    shrinkB=0,
                    shrinkA=0,
                    lw=2,
                ),
            )
            step += self.labelspacing
        # Continue from location with the positives.
        for _, value in (
            self.df.loc[self.df[f"pos_{model}"] == 1]
            .sort_values(model)
            .iterrows()
        ):
            self.ax.annotate(
                value[self.lab],
                xy=(value[model], refloc)
                if axis == "x"
                else (refloc, value[model]),
                xycoords="data",
                xytext=(start_loc + step, margin)
                if axis == "x"
                else (margin, start_loc + step),
                textcoords=trans,
                va=va,
                ha=ha,
                color=self.colors[1],
                rotation=rotation,
                size=self.labelsize,
            )
            # Little trick here to actually attach to the left center point.
            self.ax.annotate(
                "",
                xy=(value[model], refloc)
                if axis == "x"
                else (refloc, value[model]),
                xycoords="data",
                xytext=(start_loc + step, margin)
                if axis == "x"
                else (margin, start_loc + step),
                textcoords=trans,
                arrowprops=dict(
                    arrowstyle="-",
                    edgecolor=self.colors[1],
                    shrinkB=0,
                    shrinkA=0,
                    lw=2,
                ),
            )
            step += self.labelspacing

        return self

    def save(self, path, dpi=200):
        """Save figure to path."""
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(self.fig)
