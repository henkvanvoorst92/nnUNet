
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple, List, Any, Dict
from nnunetv2.my_utils.utils import rename_result_columns

def prepare_categorical_for_lineplot(
    data: pd.DataFrame,
    x: str,
    hue: Optional[str] = None,
    relabel_x: Optional[Dict[Any, str]] = None,
    dodge: float = 0.2,
    ordered: bool = True,
    suffix: str = "_plot",
    *,
    benchmark_hue: Optional[Any] = None,     # <-- stays fixed at 0 offset
    hue_order: Optional[List[Any]] = None,   # optional explicit hue order
    on_missing_benchmark: str = "error"      # "error" | "ignore" | "skip"
) -> pd.DataFrame:
    """
    Prepare a categorical x-variable (and optional hue offsets) for seaborn lineplot.

    Creates numeric x-positions for plotting (with dodge for hue) while keeping the
    benchmark hue anchored (no shift). Other hues are shifted by integer multiples
    of `dodge` relative to the benchmark's position.

    Returns a copy of `data` with:
      - f"{x}_cat" : ordered categorical version of x
      - f"{x}{suffix}" : numeric x positions (with benchmark-anchored hue offsets)
    """
    df = data.copy()

    # 1) Establish x order (and optional relabeling)
    if relabel_x:
        categories = list(relabel_x.keys())
    else:
        categories = list(pd.unique(df[x]))
    df[f"{x}_cat"] = pd.Categorical(df[x], categories=categories, ordered=ordered)

    # 2) Base numeric x codes (0..n-1)
    x_codes = df[f"{x}_cat"].cat.codes.astype(float)

    # 3) Hue-anchored offsets
    if hue is not None and dodge:
        # Determine hue levels / order
        if hue_order is not None:
            levels = [h for h in hue_order if h in pd.unique(df[hue])]
        else:
            # keep appearance order in the data
            levels = list(pd.unique(df[hue]))

        if benchmark_hue is None:
            benchmark = levels[0] if levels else None
        else:
            benchmark = benchmark_hue

        # Handle missing benchmark
        if benchmark not in levels:
            if on_missing_benchmark == "error":
                raise ValueError(
                    f"benchmark_hue='{benchmark}' not found in hue levels: {levels}"
                )
            elif on_missing_benchmark == "skip":
                # Just don’t apply any dodge if benchmark is missing
                df[f"{x}{suffix}"] = x_codes
                return df
            elif on_missing_benchmark == "ignore":
                # Proceed anyway; benchmark gets no special treatment
                pass

        # Build index mapping and anchor at benchmark
        hue_to_idx = {lvl: i for i, lvl in enumerate(levels)}
        bench_idx = hue_to_idx.get(benchmark, 0)
        # offset = (idx - bench_idx) * dodge   -> benchmark gets 0
        df[f"{x}{suffix}"] = x_codes + df[hue].map(lambda h: (hue_to_idx[h] - bench_idx) * dodge
                                                   if h in hue_to_idx else 0.0)
    else:
        df[f"{x}{suffix}"] = x_codes

    return df

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
    panel_text: Optional[List[str]] = ['A','B'],
    panel_text_kwargs: dict = dict(fontsize=16, fontweight='bold', va='top', ha='left'),
    add_grid=False
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
        if relabel_x and '_plot' not in x:
            # Map original labels to new labels; keep others unchanged
            ax.set_xticks(list(relabel_x.keys()))  # specify tick positions
            ax.set_xticklabels(list(relabel_x.values()))
        elif relabel_x:
            ax.set_xticks(list(np.arange(len(relabel_x))))
            ax.set_xticklabels(list(relabel_x.values()))

    # Use seaborn theme defaults
    sns.set(style="whitegrid")
    # then remove gridlines
    plt.rcParams['axes.grid'] = add_grid

    # Enforce ordering if relabel_x is provided
    # if relabel_x and '_plot' not in x:
    #     desired_order = list(relabel_x.keys())
    #     data = data.copy()
    #     data[x] = pd.Categorical(data[x], categories=desired_order, ordered=True)


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

        # Remove "Class =" from facet titles
        for ax in g.axes.flat:
            old_title = ax.get_title()
            if "=" in old_title:
                ax.set_title(old_title.split("=")[-1].strip())

        # Label axes / tidy up
        g.set_xlabels(x if title_x is None else title_x)

        # 1) Ensure every subplot has a y-axis label
        for ax in g.axes.flat:
            ax.set_ylabel(y)

        # 2) Re-enable y tick labels & the left spine on ALL subplots
        g.despine(left=False, right=True, top=True)  # keep left spine visible
        for ax in g.axes.flat:
            ax.set_ylabel(y, visible=True)  # show axis title
            ax.yaxis.get_label().set_visible(True)  # explicit: make label visible
            ax.tick_params(axis='y', which='both', labelleft=True)  # show tick numbers
            for ticklabel in ax.get_yticklabels():
                ticklabel.set_visible(True)

        if hue is not None:
            g._legend.set_title(hue)

        for ax in g.axes.flatten():
            _apply_relabel(ax)

        if panel_text:
            n_panels = len(g.axes.flatten())
            if len(panel_text) != n_panels:
                raise ValueError(f"panel_text length ({len(panel_text)}) "
                                 f"must match number of panels ({n_panels})")
            for ax, text in zip(g.axes.flatten(), panel_text):
                ax.text(0.02, 0.98, text, transform=ax.transAxes, **panel_text_kwargs)

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
    palette: Optional[Union[str, list]] = None,
    panel_text: Optional[List[str]] = None,
    panel_text_kwargs: dict = dict(fontsize=16, fontweight='bold', va='top', ha='left')
):
    """
    Grouped boxplots with optional faceting.
    - hue: separate boxes within each x group
    - subplot_by: separate panels (columns) per level
    """
    sns.set(style="whitegrid")

    if palette is None:
        palette = sns.color_palette()

    if subplot_by is None:
        # Single-axes grouped boxplot
        fig, ax = plt.subplots(figsize=(height * aspect, height))
        sns.boxplot(
            data=data,
            x=x, y=y,
            hue=hue,         # -> grouped boxes within each x
            ax=ax,
            palette=palette
        )
        ax.set_xlabel(x)
        ax.set_ylabel(y)

        if panel_text:
            ax.text(0.02, 0.98, panel_text[0], transform=ax.transAxes, **panel_text_kwargs)

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
            palette=palette
        )
        g.set_axis_labels(x, y)

        if hue is not None and g._legend is not None:
            g._legend.set_title(hue)

        # Flatten axes (if it's a 1‑row grid)
        axes = g.axes.flatten()
        for ax, title in zip(axes, np.unique(data[subplot_by])):
            ax.set_title(title)

        for ax in g.axes.flat:
            ax.set_ylabel(y, visible=True)  # show axis title
            ax.yaxis.get_label().set_visible(True)  # explicit: make label visible
            ax.tick_params(axis='y', which='both', labelleft=True)  # show tick numbers
            for ticklabel in ax.get_yticklabels():
                ticklabel.set_visible(True)

        if panel_text:
            axes = g.axes.flatten()
            if len(panel_text) != len(axes):
                raise ValueError(
                    f"panel_text length ({len(panel_text)}) must match number of panels ({len(axes)})"
                )
            for ax, txt in zip(axes, panel_text):
                ax.text(0.02, 0.98, txt, transform=ax.transAxes, **panel_text_kwargs)

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




def test_time_plots(data,
                          metrics=None,
                          dir_figs=None,
                          addname='test_results_',
                          select_exp=None,
                          relabel_x={0: 't-6', 1: 't-4', 2: 't-2', 3: 't=0', 4: 't+2', 5: 't+4', 6: 't+6'}
                          ):
    """
    plots metrics across generated timeframes with distribution across folds (not IDs)
    """


    data = rename_result_columns(data)
    data = prepare_categorical_for_lineplot(data, x='channel', hue='experiment',
                                            relabel_x=relabel_x, dodge=0.02)
    data_foldwise = data.groupby(['fold', 'experiment', 'channel', 'Class']).mean(numeric_only=True).reset_index()
    if select_exp is not None:
        data_select = data[np.isin(data['experiment'], select_exp)]
        data_select_foldwise = data_select.groupby(['experiment', 'channel', 'Class']).sem(numeric_only=True).reset_index()
        addname = f"{addname}{'-'.join(select_exp)}_"

    if metrics is None:
        metrics = ['Dice', 'Hausdorff', 'HD95', 'AHD', 'pred-gt_vol', 'TPR', 'PPV']
    if dir_figs is not None:
        os.makedirs(dir_figs, exist_ok=True)

    for metric in metrics:
        p_fig = os.path.join(dir_figs, f'{addname}{metric}.png') if dir_figs is not None else None

        if select_exp is not None:
            p_fig = os.path.join(dir_figs, f'{addname}{metric}.png') if dir_figs is not None else None
            lineplot_per_class(data_select, metric, 'channel_plot', 'experiment', 'Class',
                               save_path=p_fig, relabel_x=relabel_x,
                               title_x='Time to peak arterial phase (seconds)')

            p_fig = os.path.join(dir_figs, f'foldwise_{addname}{metric}.png') if dir_figs is not None else None
            lineplot_per_class(data_select_foldwise, metric, 'channel_plot', 'experiment', 'Class',
                               save_path=p_fig, relabel_x=relabel_x,
                               title_x='Time to peak arterial phase (seconds)')
        else:
            # lineplot
            lineplot_per_class(data, metric, 'channel_plot', 'experiment', 'Class',
                               save_path=p_fig, relabel_x=relabel_x,
                               title_x='Time to peak arterial phase (seconds)')

            # p_fig = os.path.join(dir_figs, f'foldwise_{addname}{metric}.png') if dir_figs is not None else None
            # lineplot_per_class(data_foldwise, metric, 'channel_plot', 'experiment', 'Class',
            #                    save_path=p_fig, relabel_x=relabel_x,
            #                    title_x='Time to peak arterial phase (seconds)')



import matplotlib as mpl


def combine_figures_with_real_axes(
        fig1, axes1,
        fig2, axes2,
        add_labels=True,
        panel_labels=("A","B","C","D"),
        label_pos=(-0.12, 1.05),
        figsize=(10, 8)
):
    """
    Combine two figures (each with 2 axes) into one 2x2 figure
    by copying plot artists (NOT images).
    """

    newfig, newaxes = plt.subplots(2, 2, figsize=figsize)

    src_axes = [axes1[0], axes1[1], axes2[0], axes2[1]]
    dst_axes = newaxes.flatten()

    for src, dst in zip(src_axes, dst_axes):

        # copy all drawable artists
        for artist in src.get_children():

            if isinstance(artist, (mpl.spines.Spine,
                                   mpl.axis.Axis,
                                   mpl.text.Text)):
                continue

            try:
                artist.remove()
                dst.add_artist(artist)
            except Exception:
                pass

        # copy axis properties
        dst.set_xlim(src.get_xlim())
        dst.set_ylim(src.get_ylim())
        dst.set_xlabel(src.get_xlabel())
        dst.set_ylabel(src.get_ylabel())
        dst.set_title(src.get_title())

        # copy ticks
        dst.set_xticks(src.get_xticks())
        dst.set_xticklabels([t.get_text() for t in src.get_xticklabels()])
        dst.set_yticks(src.get_yticks())
        dst.set_yticklabels([t.get_text() for t in src.get_yticklabels()])

        # ensure y-axis visible
        dst.tick_params(axis='y', labelleft=True)
        dst.spines["left"].set_visible(True)

    # panel labels
    if add_labels:
        for ax, lab in zip(dst_axes, panel_labels):
            ax.text(label_pos[0], label_pos[1], lab,
                    transform=ax.transAxes,
                    fontweight="bold",
                    fontsize=14)

    plt.tight_layout()
    return newfig, newaxes
def combine_figures_2x2(
        fig1, axes1,
        fig2, axes2,
        labels=("A", "B", "C", "D"),
        add_labels=True,
        label_pos=(0.02, 0.95),
        label_fontsize=14,
        figsize=(10, 8)
):
    """
    Combine two figures (each with 2 axes) into one 2x2 figure
    by rendering axes as images.

    Layout:
        axes1 -> top row
        axes2 -> bottom row
    """

    # -------- helper to convert axis to image --------
    def axis_to_image(fig, ax):
        fig.canvas.draw()

        bbox = ax.get_window_extent().transformed(
            fig.dpi_scale_trans.inverted()
        )

        width, height = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(height, width, 4)

        x0 = int(bbox.x0 * fig.dpi)
        x1 = int(bbox.x1 * fig.dpi)
        y0 = int(bbox.y0 * fig.dpi)
        y1 = int(bbox.y1 * fig.dpi)

        return buf[y0:y1, x0:x1]

    # -------- extract images --------
    imgs = [
        axis_to_image(fig1, axes1[0]),
        axis_to_image(fig1, axes1[1]),
        axis_to_image(fig2, axes2[0]),
        axis_to_image(fig2, axes2[1]),
    ]

    # -------- new figure --------
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    for ax, img in zip(axes.flatten(), imgs):
        ax.imshow(img)
        ax.axis("off")

    # -------- panel labels --------
    if add_labels:
        for ax, lab in zip(axes.flatten(), labels):
            ax.text(
                label_pos[0],
                label_pos[1],
                lab,
                transform=ax.transAxes,
                fontsize=label_fontsize,
                fontweight="bold",
                va="top"
            )

    plt.tight_layout()
    return fig, axes