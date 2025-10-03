
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union
from nnunetv2.my_utils.utils import rename_result_columns

def lineplot_per_class(
    data: pd.DataFrame,
    y: str = 'Dice',
    x: str = 'channel',
    hue: Optional[str] = 'experiment',
    subplot_by: Optional[str] = 'Class',
    errorbar: Union[str, Tuple[str, float]] = ("se", 2),
    err_style: str = "bars",
    height: float = 4.0,
    aspect: float = 1.4,
    sharey: bool = True,
    sharex: bool = True,
    save_path: Optional[str] = None,
    title_x: Optional[str] = None,
    relabel_x: Optional[dict] = None,
    ):
    """
    Line plot with optional faceting.

    Parameters
    ----------
    data : pd.DataFrame
        Long-form dataframe containing columns for x, y, and optionally hue and subplot_by.
    y, x : str
        Column names for Y and X axes.
    hue : str or None
        Column name for color grouping (separate lines/colors). If None, a single line is drawn.
    subplot_by : str or None
        Column name whose unique values define separate subplots (laid out in columns).
        If None, draws a single-axes plot.
    errorbar : str or tuple
        Seaborn-style error bar spec, e.g. "ci", "se", or ("se", 2) for 2*SE. Defaults to ("se", 2).
    err_style : {"band","bars"}
        How to render uncertainty (shaded band or discrete bars).
    height, aspect : float
        Size of each facet (inches) and width/height aspect ratio (Seaborn params).
    sharey, sharex : bool
        Whether to share y/x axes across facets.
    save_path : str or None
        If provided, saves the figure to this path.

    Returns
    -------
    fig, axes_or_grid
        Matplotlib Figure and Axes (single plot) or Seaborn FacetGrid (faceted).
    """

    def _apply_relabel(ax):
        if relabel_x:
            # Map original labels to new labels; keep others unchanged
            ax.set_xticks(list(relabel_x.keys()))  # specify tick positions
            ax.set_xticklabels(list(relabel_x.values()))
            # ticks = ax.get_xticks()
            # labels = [relabel_x.get(item.get_text(), item.get_text()) for item in ax.get_xticklabels()]
            # ax.set_xticks(ticks, labels=labels)  # preferred over set_xticklabels :contentReference[oaicite:1]{index=1}

    # Use seaborn theme defaults
    sns.set(style="whitegrid")

    if subplot_by is None:
        # Single-axes lineplot
        fig, ax = plt.subplots(figsize=(height * aspect, height))
        sns.lineplot(
            data=data,
            x=x, y=y,
            hue=hue,
            errorbar=errorbar,
            err_style=err_style,
            ax=ax
        )
        ax.set_xlabel(x if title_x is None else title_x)
        ax.set_ylabel(y)

        _apply_relabel(ax)

        if hue is not None:
            ax.legend(title=hue, frameon=True)
        else:
            # If no hue, remove legend if seaborn added any
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
        return fig, ax
    else:
        # Faceted layout: one subplot per level of `subplot_by`
        g = sns.relplot(
            data=data,
            x=x, y=y,
            hue=hue,
            kind="line",
            col=subplot_by,
            errorbar=errorbar,
            err_style=err_style,
            height=height,
            aspect=aspect,
            facet_kws={"sharey": sharey, "sharex": sharex}
        )
        # Label axes / tidy up
        g.set_xlabels(x if title_x is None else title_x)
        g.set_ylabels(y)

        if hue is not None:
            g._legend.set_title(hue)


        for ax in g.axes.flatten():
            _apply_relabel(ax)

        g.tight_layout()
        # Access underlying matplotlib Figure for saving/returning
        fig = g.figure
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
        return fig, g


def boxplot_per_class(
    data: pd.DataFrame,
    y: str = 'Dice',
    x: str = 'experiment',
    hue: Optional[str] = None,
    subplot_by: Optional[str] = 'Class',
    # kept for API symmetry but unused for boxplots:
    errorbar: Union[str, Tuple[str, float]] = ("se", 2),
    err_style: str = "bars",
    height: float = 4.0,
    aspect: float = 1.4,
    sharey: bool = True,
    sharex: bool = True,
    save_path: Optional[str] = None,
):
    """
    Grouped boxplots with optional faceting.
    - hue: separate boxes within each x group
    - subplot_by: separate panels (columns) per level
    """
    sns.set(style="whitegrid")

    if subplot_by is None:
        # Single-axes grouped boxplot
        fig, ax = plt.subplots(figsize=(height * aspect, height))
        sns.boxplot(
            data=data,
            x=x, y=y,
            hue=hue,         # -> grouped boxes within each x
            ax=ax
        )
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if hue is not None:
            ax.legend(title=hue, frameon=True)
        else:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
        return fig, ax

    else:
        # Faceted layout: one subplot per level of `subplot_by`
        g = sns.catplot(
            data=data,
            x=x, y=y,
            hue=hue,
            kind="box",      # <- figure-level API with FacetGrid
            col=subplot_by,
            height=height,
            aspect=aspect,
            sharey=sharey,
            sharex=sharex,
        )
        g.set_axis_labels(x, y)

        if hue is not None and g._legend is not None:
            g._legend.set_title(hue)

        # Flatten axes (if it's a 1‑row grid)
        axes = g.axes.flatten()
        for ax, title in zip(axes, np.unique(data[subplot_by])):
            ax.set_title(title)

        g.tight_layout()
        fig = g.figure
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)

        return fig, g


def all_val_plots(data, metrics=None, dir_figs=None, addname='val_results_', select_exp=None):
    """
    plots metrics across generated timeframes with distribution across folds (not IDs)
    """


    data = rename_result_columns(data)
    data_foldwise = data.groupby(['fold', 'experiment', 'channel', 'Class']).mean(numeric_only=True).reset_index()
    if select_exp is not None:
        data_select = data[np.isin(data['experiment'], select_exp)]
        data_select_foldwise = data_select.groupby(['experiment', 'channel', 'Class']).sem(numeric_only=True).reset_index()
        addname = f"{addname}{'-'.join(select_exp)}_"

    relabel_x = {0:'t-6', 1:'t-4', 2:'t-2', 3:'t=0',
                         4:'t+2', 5:'t+4', 6:'t+6'}

    if metrics is None:
        metrics = ['Dice', 'Hausdorff', 'HD95', 'AHD', 'pred-gt_vol', 'TPR', 'PPV']
    if dir_figs is not None:
        os.makedirs(dir_figs, exist_ok=True)

    for metric in metrics:
        p_fig = os.path.join(dir_figs, f'{addname}{metric}.png') if dir_figs is not None else None
        #lineplot
        lineplot_per_class(data, metric, 'channel', 'experiment', 'Class',
                           save_path=p_fig, relabel_x=relabel_x,
                           title_x='Time to peak arterial phase (seconds)')

        p_fig = os.path.join(dir_figs, f'foldwise_{addname}{metric}.png') if dir_figs is not None else None
        lineplot_per_class(data_foldwise, metric, 'channel', 'experiment', 'Class',
                           save_path=p_fig, relabel_x=relabel_x,
                           title_x='Time to peak arterial phase (seconds)')

        if select_exp is not None:
            p_fig = os.path.join(dir_figs, f'{addname}{metric}.png') if dir_figs is not None else None
            lineplot_per_class(data_select, metric, 'channel', 'experiment', 'Class',
                               save_path=p_fig, relabel_x=relabel_x,
                               title_x='Time to peak arterial phase (seconds)')

            p_fig = os.path.join(dir_figs, f'foldwise_{addname}{metric}.png') if dir_figs is not None else None
            lineplot_per_class(data_select_foldwise, metric, 'channel', 'experiment', 'Class',
                               save_path=p_fig, relabel_x=relabel_x,
                               title_x='Time to peak arterial phase (seconds)')








