import os
import itertools
import pandas as pd
import ast
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Tuple
import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
from scipy.stats import ttest_1samp
from nnunetv2.my_utils.utils import init_args, update_args_with_yaml, load_yaml_config, \
    get_nnUNet_paths, get_experiments, NiftiLoader, get_path_dict, combine_excel_files, np2sitk, write_multitab_excel
from nnunetv2.run.multichan_val import main_processor, main_results_processor
from nnunetv2.my_utils.plots import boxplot_per_class, test_time_plots
from nnunetv2.my_utils.tables import mean_sd_table
from nnunetv2.my_utils.stats import asterix_p_value, fit_diff_time_mixedlm, get_average_over_time
from nnunetv2.my_utils.metrics import comparative_stats, compare_multiclass_masks, compare_masks
from nnunetv2.my_utils.utils import np2sitk, image_or_path_load, sitk_dilate_mm, select_from_dataframe
from matplotlib.colors import to_rgb
from typing import List, Optional

def test_figures(pc, args, channel_dct=None, select_exp=['t246','t0']):


    if channel_dct is None:
        channel_dct = {'m6':'-6s', 'm4':'-4s', 'm2':'-2s', 't0':'0s', 'p2':'+2s', 'p4':'+4s', 'p6':'+6s'}

    pc = pc[np.isin(pc['channel'], list(channel_dct.keys())+['cta'])]

    dir_fig_cta = os.path.join(args.p_out, 'figures', 'cta')
    os.makedirs(dir_fig_cta, exist_ok=True)
    dir_fig_4d = os.path.join(args.p_out, 'figures', '4d')
    os.makedirs(dir_fig_4d, exist_ok=True)

    #rename classes for plot
    pc['Class'] = pc['Class'].map({1: 'Artery', 2: 'Vein', 3: 'Both'})
    #pc = pc[(pc['channel'] == 'cta') & np.isin(pc['experiment'], select_exp)]
    pc = pc.rename(columns={'Dice':'Dice Similarity Coefficient (DSC)'})

    test_time_plots(pc[pc['channel']!='cta'],
                    dir_figs=dir_fig_4d,
                    addname='test_results_',
                    select_exp=select_exp,
                    metrics=['Dice Similarity Coefficient (DSC)'],
                    relabel_x=channel_dct
                    )

    outcome = 'Dice Similarity Coefficient (DSC)'
    boxplot_per_class(pc[(pc['channel'] == 'cta') & np.isin(pc['experiment'], select_exp)],
                      y=outcome, x='experiment',
                      subplot_by='Class',
                      save_path=os.path.join(dir_fig_cta, f'boxplot_cta_{outcome}.png'),
                      palette={'t246':'#4c72b0', 't0':'#dd8452'},
                      panel_text=['C','D']
                      )


def get_test_results(p_out, overwrite=False):
    p_per_ID_res = os.path.join(p_out, 'test_results_per_ID')
    p_pic = os.path.join(p_out, 'test_results.pic')
    if not os.path.exists(p_pic) or overwrite:
        data = combine_excel_files(p_per_ID_res, 'results.xlsx')
        data.to_pickle(p_pic)
    else:
        data = pd.read_pickle(p_pic)

    return data

def build_metric_tables(
    df: pd.DataFrame,
    metrics: list[str], exp_order=None
) -> dict[str, pd.DataFrame]:
    """
    Builds tables for each metric with shape:
       index = experiment, res_type
       columns = dataset, version (w / w/o)

    Args:
        df: input DataFrame
        metrics: list of metric column names

    Returns:
        Dict of metric name -> pivoted DataFrame
    """

    df = df.copy()

    # parse model (exp_base) and version (w or w/o) from experiment
    # for example "CTFM w" => base="CTFM", version="w"
    parts = df["experiment"].str.rsplit(" ", n=1, expand=True)
    df["exp_base"] = parts[0]
    df["version"] = parts[1]

    # ensure consistent ordering of res_type if desired
    order = ["Artery", "Vein", "Any vessel", "Macro-average", "micro_avg"]
    df["res_type"] = pd.Categorical(df["res_type"], categories=order, ordered=True)

    if exp_order is not None:
        df["exp_base"] = pd.Categorical(df["exp_base"], categories=exp_order, ordered=True)

    tables = {}
    for metric in metrics:
        pivot = df.pivot_table(
            index=['analysis',"res_type","exp_base"],
            columns=["dataset", "version"],
            values=metric,
            aggfunc=lambda x: " / ".join(x.astype(str))
        )

        # sort indices and columns
        pivot = pivot.sort_index(axis=0, level=[0,1])
        pivot = pivot.sort_index(axis=1, level=[0,1])

        tables[metric] = pivot

    return tables

def compute_id_wise_differences(data: pd.DataFrame, round_dct, exp_combis: list[tuple[str, str]]) -> pd.DataFrame:
    """
    Compute ID-wise differences per experiment, per metric, per res_type, and per dataset.

    Args:
        data: Input DataFrame containing the data.
        metrics: List of metric column names to compute differences for.
        exp_combis: List of tuples, where each tuple contains two experiments to compare (e.g., [('exp1', 'exp2')]).

    Returns:
        DataFrame with ID-wise differences.
    """
    results = []

    for exp1, exp2 in exp_combis:
        # Filter data for the two experiments
        subset = data[data['experiment'].isin([exp1, exp2])]
        tmp = []
        for metric in round_dct.keys():
            # Pivot data to align metrics for each ID
            pivoted = subset.pivot_table(
                index=['main_dataset','ID', 'res_type', 'dataset','analysis'],
                columns='experiment',
                values=metric
            )

            # Compute differences for each metric
            diff = pd.DataFrame(pivoted[exp1] - pivoted[exp2])
            diff.columns = [metric]
            tmp.append(diff)
        r = pd.concat(tmp, axis=1, join='outer').reset_index()
        r['experiment_diff'] = "{} {} - {}".format(exp1.split(' ')[0], exp1.split(' ')[1], exp2.split(' ')[1])
        results.append(r)

    # Combine results for all experiment combinations
    results = pd.concat(results, ignore_index=True)
    return results

def stat_compare_differences(data: pd.DataFrame,
                             round_dct,
                             exp_col: str = 'experiment_diff',
                             groups=[],
                             ID_col: str = 'ID',
                             time_col: str = 'channel',
                             multiple_measurements_exp_col='simCTA'):
    #add addition aggregation
    if len(groups) == 0:
        groups = [ exp_col,'main_dataset', 'dataset','res_type', 'analysis']
    else:
        groups = [exp_col] + groups

    msd_res = mean_sd_table(
        data,  # Assuming `results` is the concatenated DataFrame
        partition_by=groups,
        rounding=round_dct,
        use_plus_minus=True,
        use_se=True
    ).set_index(groups)

    for metric in round_dct.keys():
        msd_res[f'{metric}_tstat'] = np.nan
        msd_res[f'{metric}_pvalue'] = np.nan

    for group_key, gd in data.groupby(groups):
        for metric, rnd in round_dct.items():
            t, p = ttest_1samp(gd[metric], popmean=0)
            msd_res.at[group_key, f'{metric}_tstat'] = t
            msd_res.at[group_key, f'{metric}_pvalue'] = p
            pstar = asterix_p_value(p)
            msd_res.at[group_key, metric] += pstar

    #make additional results for simCTA across time difference
    #if multiple_measurements_exp_col in data.columns:
    mm_data = data[data['main_dataset']=='stanford']
    if len(mm_data)>0:
        mm_groups = groups
        if 'dataset' in mm_groups:
            mm_groups.remove('dataset')

        #mixedeffects per model for differences across all timepoints
        mm_out = []
        for group_key, gd in tqdm(mm_data.groupby(mm_groups), desc='MixedLM for time differences'):
            for metric, rnd in round_dct.items():
                try:
                    model = fit_diff_time_mixedlm(gd, id_col=ID_col, time_col=time_col, diff_col=metric, reml=True)
                    res = get_average_over_time(model, time_col=time_col)
                except Exception as e:
                    print(f"Error fitting mixedlm for group {group_key}, metric {metric}: \n{e}")
                    continue
                row = list(group_key) + [metric, multiple_measurements_exp_col, len(gd)] + list(res.values())
                #make string mean ±se with asterix for pvalue
                pstar = asterix_p_value(res['p_value'])
                mn = round(res['mean_diff'], rnd) if rnd>0 or np.isnan(res['mean_diff']) else int(round(res['mean_diff'], rnd))
                se = round(res['se'], rnd) if rnd>0 or np.isnan(res['se']) else int(round(res['se'], rnd))
                meanse_str = f"{mn} ±{se}{pstar}"
                row.append(meanse_str)
                mm_out.append(row)
            mm_cols = list(mm_groups) + ['metric', 'mm_exp', 'n_IDs'] + list(res.keys()) + ['mean±se']
            mm_df = pd.DataFrame(mm_out, columns=mm_cols)

    return msd_res.reset_index(), mm_df


def lineplot_multi_outcomes(
        data: pd.DataFrame,
        ys: List[str],  # <--- list of outcome columns
        x: str = 'channel',
        hue: Optional[str] = 'experiment',  # grouping (used for color)
        outcome_hue: bool = False,  # whether to color by outcome as well
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
        panel_text: Optional[List[str]] = ['A', 'B'],
        subplot_text: Optional[List[str]] = None,
        panel_text_kwargs: dict = dict(fontsize=16, fontweight='bold', va='top', ha='left'),
        subplot_text_kwargs: dict = dict(fontsize=10, color='black'), #, va='bottom', ha='right'
        add_grid=False
):
    """Line plot with multiple outcome (y) series in the same figure."""

    # melt to long form with an “Outcome” indicator
    plot_data = data.melt(
        id_vars=[x] + ([hue] if hue else []) + ([subplot_by] if subplot_by else []),
        value_vars=ys,
        var_name='Outcome',
        value_name='y_val'
    )

    # If we want “Outcome” itself in the legend/color, combine with hue
    if outcome_hue and hue:
        plot_data['hue_combo'] = plot_data[hue].astype(str) + "_" + plot_data['Outcome']
        hue_arg = 'hue_combo'
    elif outcome_hue:
        hue_arg = 'Outcome'
    else:
        hue_arg = hue

    if hue_arg:
        unique_hues = plot_data[hue_arg].unique()
        if len(unique_hues) > 1:
            # Apply offset to the second line (or any specific line)
            plot_data.loc[plot_data[hue_arg] == unique_hues[1], x] += 0.1

    # Use seaborn theme defaults; ensure grid on/off
    sns.set(style="whitegrid")
    plt.rcParams['axes.grid'] = add_grid

    # base common labelling
    xlabel = x if title_x is None else title_x
    ylabel = "Value"

    if subplot_by is None:
        # simple multi-line single axes
        fig, ax = plt.subplots(figsize=(height * aspect, height))
        sns.lineplot(
            data=plot_data,
            x=x, y='y_val',
            hue=hue_arg,
            errorbar=errorbar,
            err_style=err_style,
            ax=ax
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if relabel_x:
            ax.set_xticks(list(relabel_x.keys()))
            ax.set_xticklabels(list(relabel_x.values()))

        if hue_arg:
            ax.legend(title=hue_arg)
        else:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        ax.get_legend().remove()

        if subplot_text:
            # only one subplot here, so take first text
            ax.text(
                0.08, 0.04, subplot_text[0],
                transform=ax.transAxes,
                **subplot_text_kwargs
            )

        #fig.tight_layout(pad=1.0, h_pad=1.0, w_pad=1.0)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
        return fig, ax

    else:
        # faceted by subplot_by
        g = sns.relplot(
            data=plot_data,
            x=x, y='y_val',
            hue=hue_arg,
            kind="line",
            row='Outcome',
            col=subplot_by,
            errorbar=errorbar,
            err_style=err_style,
            height=height,
            aspect=aspect,
            facet_kws={"sharey": sharey, "sharex": sharex}
        )

        # clean facet titles
        for ax in g.axes.flat:
            title = ax.get_title()
            if "=" in title:
                ax.set_title(title.split("=")[-1].strip())
            ax.tick_params(axis='y', labelleft=True)

        g.set_xlabels(xlabel)
        for i, outcome in enumerate(g.row_names):
            for j in range(len(g.col_names)):
                ax = g.axes[i][j]
                ax.set_ylabel(outcome)
                ax.yaxis.set_tick_params(labelleft=True)


        if not sharey:
            for i, row_axes in enumerate(g.axes):
                y_min, y_max = float('inf'), float('-inf')
                for ax in row_axes:
                    y_min = min(y_min, ax.get_ylim()[0])
                    y_max = max(y_max, ax.get_ylim()[1])
                for ax in row_axes:
                    ax.set_ylim(y_min, y_max)

        # optional relabel_x
        if relabel_x:
            for ax in g.axes.flat:
                ax.set_xticks(list(relabel_x.keys()))
                ax.set_xticklabels(list(relabel_x.values()))

        # add panel text if requested
        if panel_text:
            axes = g.axes.flatten()
            if len(panel_text) != len(axes):
                raise ValueError("panel_text length must match number of panels")
            for ax, txt in zip(axes, panel_text):
                ax.text(0.02, 0.98, txt, transform=ax.transAxes, **panel_text_kwargs)

        if subplot_text:
            axes = g.axes.flatten()
            if len(subplot_text) != len(axes):
                raise ValueError("subplot_text length must match number of panels")
            for ax, txt in zip(axes, subplot_text):
                ax.text(
                    0.08, 0.04, txt,
                    transform=ax.transAxes,
                    **subplot_text_kwargs
                )

        if g._legend is not None:
            g._legend.remove()



        handles, labels = g.axes.flatten()[0].get_legend_handles_labels()
        leg = g.fig.legend(
            handles, labels,
            loc="upper right",
            bbox_to_anchor=(0.98, 1.0),
            title=None
        )
        # remove frame
        leg.get_frame().set_linewidth(0.0)
        leg.get_frame().set_facecolor("none")
        leg.get_frame().set_edgecolor("none")

        fig = g.figure
        fig.tight_layout(pad=1.0, h_pad=1.0, w_pad=1.0)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
            fig.savefig(save_path.replace('.png', '.svg'), dpi=300)

        return fig, g

def channel_cols(data, select_chans=None):
    chs, chn, scta = [], [], []
    for __,row in data.iterrows():
        ds = row['dataset']
        chan_str = ds if row['main_dataset']=='stanford' else None
        if chan_str is not None:
            chan_num = int(chan_str[1:])*-1 if 'm' in chan_str else int(chan_str[1:])
        else:
            chan_num = chan_str
        chs.append(chan_str)
        chn.append(chan_num)
    data['channel_name'] = chs
    data['channel'] = chn
    if select_chans is not None:
        data = data[(np.isin(data['channel_name'], select_chans))|(data['channel_name'].isna())]

    return data

def get_main_dataset_name(IDs):
    out = []
    for ID in IDs:
        if 'SU0'== str(ID)[:3]:
            out.append('stanford')
        elif len(str(ID))==4:
            out.append('cta')
        else:
            out.append('poorcta')
    return out

def data_prep(data):
    data['FPR'] *= 1000 #show FPR per 1000 as it is very small
    data['res_type'] =data['res_type'].replace({'macro_avg':'Macro-average', 'micro-avg':'Micro-average', 1:'Artery', 2:'Vein', 0:'Any vessel'})
    data['main_dataset'] =  get_main_dataset_name(data.ID)
    data['experiment'] = [exp.replace('time_averages_', 'cta_') for exp in data['experiment']]
    data['experiment'] = data['experiment'].replace(args.experiments)
    data['model'] = [exp.split(' ')[0] if ' w' in exp else exp for exp in data['experiment']]
    data['model_subtype'] = [exp.split(' ')[1] if ' w' in exp else '' for exp in data['experiment']]

    data = channel_cols(data, select_chans=getattr(args, 'chans', None))
    return data

def fetch_simcta_stat(stat_table, ediff, ds, outcome, res_type, return_val=None, string_return=''):


    perf_res = stat_table[(stat_table['experiment_diff'] == ediff) &
                          (stat_table['main_dataset'] == 'stanford') &
                           (stat_table['analysis'] == ds) &
                           (stat_table['res_type'] == res_type)]

    # return_val should be column in stat_table
    if return_val is not None:
        val = perf_res[perf_res['metric'] == outcome][return_val].values[0].replace(' ','')
        out = f'{string_return}{val}'
    else:
        out = perf_res
    return out

def simcta_plots(data, exp_combis, chan_dct, time_stat,
                 outcomes=['Dice', 'clDice', 'HD95', 'AVD'],
                 res_types=['Artery', 'Vein'], dir_figures=None):

    subplot_id = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

    data = data[data['main_dataset']=='stanford']

    for ds in data['analysis'].unique():
        if ds is None:
            continue
        for (e1, e2) in exp_combis:
            # cols artery-vein, rows: Dice, clDice, HD95, Betti0
            #two lines for performance (e1 and e2)
            sdct = {'experiment': [e1, e2],
                   'channel': list(chan_dct.values()),
                   'analysis': ds,
                   'res_type':res_types}
            tmp = select_from_dataframe(data, conditions_dict=sdct)
            ediff = f"{e1.split(' ')[0]} {e1.split(' ')[1]} - {e2.split(' ')[1]}"

            #add stat res in plot for comparison
            perf_diffs = []
            for outcome in outcomes:
                for res_type in res_types:
                    plot_str = fetch_simcta_stat(time_stat, ediff, ds, outcome, res_type,
                                            return_val='mean±se', string_return='diff: ')
                    perf_diffs.append(plot_str)

            n_subplots = len(tmp['res_type'].unique())*len(outcomes)
            #make plot for all outcomes
            ed = ediff.replace(os.sep,'')
            lineplot_multi_outcomes(tmp,
                                    ys=outcomes, x='channel',
                                    hue='experiment', subplot_by = 'res_type',
                                    panel_text = [subplot_id[i] for i in range(n_subplots)],
                                    relabel_x={v:v for v in chan_dct.values()},
                                    sharey=False,sharex=False,
                                    title_x='Time to peak arterial (t=0) in seconds',
                                    subplot_text=perf_diffs,
                                    save_path=os.path.join(dir_figures,f'simcta_{ds}_{ed}.png') if dir_figures is not None else None
                                    )
            plt.show()



def adjust_color(color, amount=0.25):
    c = np.array(to_rgb(color), dtype=float)
    if amount > 0:
        # darken toward black
        c = c * (1 - amount)
    elif amount < 0:
        # lighten toward white
        c = c + (1 - c) * (-amount)
    return tuple(np.clip(c, 0, 1))


def barplot_performance_models(
    data: pd.DataFrame,
    outcomes: List[str],
    model: str = "model",
    model_subtype: Optional[str] = None,
    model_label: str = "experiment",
    res_type: str = "res_type",
    height: float = 3.5,
    aspect: float = 1.4,
    bar_order: Optional[List[str]] = None,
    row_order: Optional[List[str]] = None,
    col_order: Optional[List[str]] = None,
    sharey: bool = False,
    sharey_rows: bool = True,
    title: Optional[str] = None,
    palette: str = "tab10",
    hatches: Optional[dict] = None,
    vline_after: Optional[List[str]] = None,
    vline_after_idx: Optional[List[int]] = None,
    vline_kwargs: Optional[dict] = None,
    vline_annotations: Optional[List[dict]] = None,
    between_bar_marks: Optional[List[dict]] = None,
):
    if vline_kwargs is None:
        vline_kwargs = dict(color="k", linestyle="--", linewidth=1, alpha=0.6)

    df = data.copy()

    # -----------------------------
    # reshape outcomes → rows
    # -----------------------------
    id_vars = [model, res_type]
    if model_subtype is not None and model_subtype in df.columns:
        id_vars.append(model_subtype)
    if model_label is not None:
        id_vars.append(model_label)

    df = df.melt(
        id_vars=id_vars,
        value_vars=outcomes,
        var_name="Outcome",
        value_name="Value"
    )

    if row_order is not None:
        df["Outcome"] = pd.Categorical(
            df["Outcome"], categories=row_order, ordered=True
        )
    else:
        row_order = df["Outcome"].unique().tolist()

    if col_order is not None:
        df[res_type] = pd.Categorical(
            df[res_type], categories=col_order, ordered=True
        )
    else:
        col_order = df[res_type].unique().tolist()
    # -----------------------------
    # composite model label
    # -----------------------------

    if bar_order is not None:
        df[model_label] = pd.Categorical(
            df[model_label], categories=bar_order, ordered=True
        )
    else:
        bar_order = df[model_label].unique().tolist()

    # -----------------------------
    # palette per experiment
    # -----------------------------
    if isinstance(palette, str):
        palette = dict(
            zip(
                df[model_label].unique(),
                sns.color_palette(palette, df[model_label].nunique())
            )
        )

    # -----------------------------
    # main plot
    # -----------------------------
    g = sns.catplot(
        data=df,
        x=model_label,
        y="Value",
        hue=model_label,        # color only
        col=res_type,
        row="Outcome",
        kind="bar",
        errorbar="se",
        order=bar_order,
        height=height,
        aspect=aspect,
        palette=palette,
        sharey=sharey,
        legend=False,
    )

    if hatches is not None:
        idx_to_label = {i: label for i, label in enumerate(bar_order)}

        for ax in g.axes.flat:
            # number of bars per group and number of categories
            n_bars = len(ax.patches)
            n_labels = len(bar_order)

            for i, bar in enumerate(ax.patches):
                # find the category index this bar corresponds to
                # (bars are drawn in order of categories repeated by column/row facets)
                label_idx = i % n_labels
                label = idx_to_label[label_idx]

                # set hatch from map
                hatch = hatches.get(label, "")
                bar.set_hatch(hatch)

    g.figure.canvas.draw()
    # -----------------------------
    # titles and y-labels
    # -----------------------------
    for j, col_name in enumerate(g.col_names):
        for i in range(len(g.row_names)):
            g.axes[i][j].set_title(str(col_name))

    for i, outcome in enumerate(g.row_names):
        g.axes[i][0].set_ylabel(outcome)
    # -----------------------------
    # remove default x ticks
    # -----------------------------
    bottom_axes = g.axes[-1]
    bar_labels = [t.get_text() for t in bottom_axes[0].get_xticklabels()]

    for ax in g.axes.flat:
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks([])
        ax.set_xlabel("")

    # -----------------------------
    # draw DIAGONAL labels UNDER x-axis (robust)
    # -----------------------------
    for i, outcome in enumerate(g.row_names):
        for j, col_name in enumerate(g.col_names):
            ax = g.axes[i][j]

            sub = df[
                (df["Outcome"] == outcome) &
                (df[res_type].astype(str) == str(col_name))
            ]

            labels = bar_labels
            patches = ax.patches

            if len(labels) != len(patches):
                labels = labels[:len(patches)]

            bar_positions = sorted(
                [patch.get_x() + patch.get_width() / 2 for patch in ax.patches]
            )
            xlim = ax.get_xlim()
            for patch, label, x_center in zip(patches, labels, bar_positions):
                x_frac = (x_center - xlim[0]) / (xlim[1] - xlim[0])

                ax.text(
                    x_frac,
                    -0.03,             # BELOW x-axis
                    label,
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    rotation=45,           # diagonal
                    fontsize=8,
                    clip_on=False
                )

    for i, outcome in enumerate(g.row_names):
        for j, col_name in enumerate(g.col_names):
            ax = g.axes[i][j]

            sub = df[
                (df["Outcome"] == outcome) &
                (df[res_type].astype(str) == str(col_name))
            ]

            if sub.empty:
                continue

            # compute mean and SE per bar (same logic as seaborn)
            stats = (
                sub
                .groupby(model_label, observed=True)["Value"]
                .agg(
                    mean="mean",
                    se=lambda x: x.std(ddof=1) / np.sqrt(len(x))
                )
                .reindex(bar_order)      # ensure bar order consistency
                .dropna()
            )

            if stats.empty:
                continue

            lower_ci = stats["mean"] - stats["se"]
            upper_ci = stats["mean"] + stats["se"]

            worst_lower = lower_ci.min()
            best_upper = upper_ci.max()

            # handle degenerate case
            if worst_lower == best_upper:
                eps = 1e-9 if best_upper == 0 else abs(best_upper) * 1e-9
                ymin = worst_lower - eps
                ymax = best_upper + eps
            else:
                ymin = worst_lower * 0.95
                ymax = best_upper * 1.05

            ax.set_ylim(ymin, ymax)


    # -----------------------------
    # standardize y-axis limits per row
    # -----------------------------
    if sharey_rows:

        for i, outcome in enumerate(g.row_names):

            ymin_all = np.inf
            ymax_all = -np.inf

            # first pass: collect limits across columns
            for j, col_name in enumerate(g.col_names):
                ax = g.axes[i][j]

                sub = df[
                    (df["Outcome"] == outcome) &
                    (df[res_type].astype(str) == str(col_name))
                    ]

                if sub.empty:
                    continue

                stats = (
                    sub
                    .groupby(model_label, observed=True)["Value"]
                    .agg(
                        mean="mean",
                        se=lambda x: x.std(ddof=1) / np.sqrt(len(x))
                    )
                    .reindex(bar_order)
                    .dropna()
                )

                if stats.empty:
                    continue

                lower = (stats["mean"] - stats["se"]).min()
                upper = (stats["mean"] + stats["se"]).max()

                ymin_all = min(ymin_all, lower)
                ymax_all = max(ymax_all, upper)*1.07

            if not np.isfinite(ymin_all) or not np.isfinite(ymax_all):
                continue

            # padding
            if ymin_all == ymax_all:
                eps = 1e-9 if ymin_all == 0 else abs(ymin_all) * 1e-9
                ymin = ymin_all - eps
                ymax = ymax_all + eps
            else:
                ymin = ymin_all - 0.05 * (ymax_all - ymin_all)
                ymax = ymax_all + 0.05 * (ymax_all - ymin_all)

            # second pass: apply limits to entire row
            for j in range(len(g.col_names)):
                g.axes[i][j].set_ylim(ymin, ymax)

    # -----------------------------
    # vertical dashed lines AFTER specified bars only
    # -----------------------------
    if vline_after is not None or vline_after_idx is not None:

        if vline_kwargs is None:
            vline_kwargs = dict(
                color="k", linestyle="--", linewidth=0.8, alpha=0.4
            )

        for ax in g.axes.flat:
            patches = ax.patches
            if len(patches) <= 1:
                continue

            # bar centers in plotting order
            centers = [
                p.get_x() + p.get_width() / 2 for p in patches
            ]

            # resolve indices where to draw vlines
            if vline_after is not None:
                after_idx = [
                    bar_order.index(lbl)
                    for lbl in vline_after
                    if lbl in bar_order
                ]
            else:
                after_idx = vline_after_idx

            # draw vline between bar i and i+1
            for idx in after_idx:
                if idx < 0 or idx >= len(centers) - 1:
                    continue

                x = 0.5 * (centers[idx] + centers[idx + 1])
                ax.axvline(x, **vline_kwargs)

    # -----------------------------
    # panel annotations: top-aligned under ymax, facet-aware (row, col)
    # -----------------------------
    if vline_annotations is not None:

        for i, outcome in enumerate(g.row_names):
            for j, col_name in enumerate(g.col_names):
                ax = g.axes[i][j]

                ymin, ymax = ax.get_ylim()
                yrange = ymax - ymin

                for ann in vline_annotations:

                    # facet filtering
                    # facet = ann.get("facet")
                    # if facet is not None:
                    #     if isinstance(facet, tuple):
                    #         facet = [facet]
                    #     if (outcome, col_name) not in facet:
                    #         continue
                    # else:
                    #     continue
                    # y just under ymax (data coords)
                    y = ymax - ann.get("y_offset", 0.02) * yrange

                    # left-anchored text
                    if ann.get("x") == "left":
                        ax.text(
                            0.01,
                            y,
                            ann["text"],
                            transform=ax.get_yaxis_transform(),
                            **ann.get("text_kwargs", {})
                        )
                        continue

                    # index-based x position
                    patches = ax.patches
                    if not patches:
                        continue

                    centers = [
                        p.get_x() + p.get_width() / 2 for p in patches
                    ]

                    idx = ann.get("after_idx")
                    if idx is None or idx < 0 or idx >= len(centers):
                        continue

                    x = centers[idx] + ann.get("dx", 0.0)

                    ax.text(
                        x,
                        y,
                        ann["text"],
                        **ann.get("text_kwargs", {})
                    )

    # -----------------------------
    # marks between bars (facet-aware via (row, col))
    # -----------------------------
    if between_bar_marks is not None:

        for i, outcome in enumerate(g.row_names):
            for j, col_name in enumerate(g.col_names):
                ax = g.axes[i][j]

                patches = ax.patches
                if len(patches) < 2:
                    continue

                centers = [
                    p.get_x() + p.get_width() / 2 for p in patches
                ]

                ymin, ymax = ax.get_ylim()
                yrange = ymax - ymin

                for mark in between_bar_marks:

                    # facet filtering
                    facet = mark.get("facet")
                    if facet is not None:
                        if isinstance(facet, tuple):
                            facet = [facet]
                        if (outcome, col_name) not in facet:
                            continue
                    else:
                        continue

                    idx = mark.get("after_idx")
                    if idx is None or idx < 0 or idx >= len(centers) - 1:
                        continue

                    x = 0.5 * (centers[idx] + centers[idx + 1])
                    x += mark.get("dx", 0.0)

                    y = ymax - mark.get("y_offset", 0.03) * yrange

                    ax.text(
                        x,
                        y,
                        mark.get("mark", "*"),
                        **mark.get("text_kwargs", {})
                    )

    if title:
        g.figure.suptitle(title, fontsize=14)
        g.figure.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        g.figure.tight_layout()

    return g

def barplots():


    analyses_together = {'minCC0.01mL':['headmasks_minCC0.01mL', 'headmasks_adj509_dil5_minCC0.01mL'],
                         'minCC0mL': ['headmasks_minCC1e-07mL', 'headmasks_adj509_dil5_minCC1e-07mL'],
                         }

    #colors = {k:v['color'] for k,v in args.plot_info.items()}
    #hatches = {k:v['hatch'] for k,v in args.plot_info.items()}
    colors = args.colors
    hatches = None

    for name, analysis in analyses_together.items():
        tmp = data.copy()
        tmp['dataset'] = tmp['dataset'].replace(args.chans, 'ctp_sim')
        #tmp['dataset'] = tmp['dataset'].replace('t0', 'ctp_sim')
        sdct = {'res_type': res_types,
                'dataset': ['ctp_sim','cta', 'poorcta'],
                'analysis':analysis}

        tmp = select_from_dataframe(tmp, conditions_dict=sdct)
        tmp['dataset'] = tmp['dataset'].replace('ctp_sim', 'Simulated CTA').replace('cta', 'Real CTA').replace(
            'poorcta', 'Poor quality CTA')
        col_order = ['Simulated CTA', 'Real CTA', 'Poor quality CTA']

        vline_annotations = [
            dict(
                text="From scratch",
                x="left",  # special keyword → left margin
                y_offset=0.01,
                text_kwargs=dict(
                    fontsize=9,
                    ha="left",
                    va="top",
                    alpha=0.8,
                ),
            ),
            dict(
                after_idx=6,  # position next to this split
                text="Finetuned FM",
                dx=0.01,  # small horizontal offset
                y_offset=0.01,
                text_kwargs=dict(
                    fontsize=9,
                    ha="left",
                    va="top",
                    alpha=0.8,
                ),
            ),
        ]

        between_bar_marks = [
            dict(
                after_idx=10,  # between bar 1 and 2
                mark="*",  # any text: "*", "†", "ns", "●"
                #rows=["Dice", 'clDice'],
                #cols=['Simulated CTA', 'Real CTA', 'Poor quality CTA'],
                facet=[('Dice', 'Simulated CTA'), ('Dice', 'Real CTA'), ('Dice', 'Poor quality CTA'),
                       ('clDice', 'Simulated CTA'), ('clDice', 'Real CTA'), ('clDice', 'Poor quality CTA'),
                       ('AVD', 'Simulated CTA'), ('AVD', 'Real CTA'), ('AVD', 'Poor quality CTA'),
                       ],
                y_offset=0.1,  # fraction below ymax
                dx=0.0,  # optional horizontal tweak
                text_kwargs=dict(
                    fontsize=9,
                    ha="center",
                    va="bottom",
                ),
            ),
            dict(
                after_idx=12,  # between bar 1 and 2
                mark="**",  # any text: "*", "†", "ns", "●"
                facet=[('Dice', 'Simulated CTA'), ('Dice', 'Real CTA'), ('Dice', 'Poor quality CTA'),
                       ('clDice', 'Simulated CTA'), ('clDice', 'Real CTA'), ('clDice', 'Poor quality CTA'),
                       ('AVD', 'Simulated CTA'), ('AVD', 'Real CTA'), ('AVD', 'Poor quality CTA'),
                       ],
                y_offset=0.1,  # fraction below ymax
                dx=0.0,  # optional horizontal tweak
                text_kwargs=dict(
                    fontsize=9,
                    ha="center",
                    va="bottom",
                ),
            )
        ]

        barplot_performance_models(
                                    tmp[tmp['res_type']=='Artery'],
                                    outcomes,  # rows for performance measures
                                    title='Artery segmentation performance',
                                    model = "model",  # model type (color) --> main hue
                                    model_subtype = "model_subtype",  # lighter shade --> sub hue
                                    model_label = 'experiment',
                                    res_type = "dataset",#"res_type",  # columns --> artery-vein-any vessel or dataset
                                    bar_order = list(args.experiments.values()),
                                    palette = colors,
                                    hatches = hatches,
                                    col_order = col_order,
                                    sharey=False,
                                    sharey_rows=True,
                                    vline_after_idx=[5,13],
                                    vline_annotations=vline_annotations,
                                    between_bar_marks=between_bar_marks,
                                    )
        plt.savefig(os.path.join(dir_figures, f'{name}_performance_barplot_artery.tiff'), bbox_inches="tight", dpi=300)
        plt.savefig(os.path.join(dir_figures, f'{name}_performance_barplot_artery.png'))
        plt.show()


        between_bar_marks = [
            dict(
                after_idx=10,  # between bar 1 and 2
                mark="*",  # any text: "*", "†", "ns", "●"
                facet=[('Dice', 'Simulated CTA'), ('Dice', 'Real CTA'), ('Dice', 'Poor quality CTA'),
                        ('clDice', 'Real CTA'), ('clDice', 'Poor quality CTA'),
                       ('AVD', 'Simulated CTA'), ('AVD', 'Real CTA'), ('AVD', 'Poor quality CTA'),
                       ],
                y_offset=0.1,  # fraction below ymax
                dx=0.0,  # optional horizontal tweak
                text_kwargs=dict(
                    fontsize=9,
                    ha="center",
                    va="bottom",
                ),
            ),
            dict(
                after_idx=12,  # between bar 1 and 2
                mark="**",  # any text: "*", "†", "ns", "●"
                facet=[('Dice', 'Simulated CTA'), ('Dice', 'Real CTA'), ('Dice', 'Poor quality CTA'),
                       ('clDice', 'Simulated CTA'), ('clDice', 'Real CTA'), ('clDice', 'Poor quality CTA'),
                       ('AVD', 'Simulated CTA'), ('AVD', 'Real CTA'), ('AVD', 'Poor quality CTA'),
                       ],
                y_offset=0.1,  # fraction below ymax
                dx=0.0,  # optional horizontal tweak
                text_kwargs=dict(
                    fontsize=9,
                    ha="center",
                    va="bottom",
                ),
            )
        ]

        barplot_performance_models(
                                    tmp[(tmp['res_type']=='Vein') & (tmp['experiment']!='Canals et al.')],
                                    outcomes,  # rows for performance measures
                                    title='Vein segmentation performance',
                                    model = "model",  # model type (color) --> main hue
                                    model_subtype = "model_subtype",  # lighter shade --> sub hue
                                    model_label = 'experiment',
                                    res_type = "dataset",#"res_type",  # columns --> artery-vein-any vessel or dataset
                                    bar_order = list(args.experiments.values()),
                                    palette = colors,
                                    hatches = hatches,
                                    col_order = col_order,
                                    sharey=False,
                                    sharey_rows=True,
                                    vline_after_idx=[5, 13],
                                    vline_annotations=vline_annotations,
                                    between_bar_marks=between_bar_marks,
                                    )
        plt.savefig(os.path.join(dir_figures, f'{name}_performance_barplot_vein.tiff'), bbox_inches="tight", dpi=300)
        plt.savefig(os.path.join(dir_figures, f'{name}_performance_barplot_vein.png'))
        plt.show()

    # barplot_performance_models(
    #                             tmp[tmp['res_type']=='Any vessel'],
    #                             outcomes,  # rows for performance measures
    #                             title='Any vessel (artery or vein) segmentation performance',
    #                             model = "model",  # model type (color) --> main hue
    #                             model_subtype = "model_subtype",  # lighter shade --> sub hue
    #                             model_label = 'experiment',
    #                             res_type = "dataset",#"res_type",  # columns --> artery-vein-any vessel or dataset
    #                             bar_order = list(args.experiments.values()),
    #                             palette = colors,
    #                             col_order = ['CTP peak arterial (t=0)', 'Real CTA', 'Poor quality CTA'],
    #                             sharey=False,
    #                             )
    # plt.savefig(os.path.join(dir_figures, 'performance_barplot_anyvessel.tiff'), bbox_inches="tight", dpi=300)
    # plt.savefig(os.path.join(dir_figures, 'performance_barplot_anyvessel.png'))
    # plt.show()

def diff_plots(data):

    analyses_together = {'minCC0.01mL':['headmasks_minCC0.01mL', 'headmasks_adj509_dil5_minCC0.01mL'],
                         'minCC0mL': ['headmasks_minCC1e-07mL', 'headmasks_adj509_dil5_minCC1e-07mL'],
                         }

    #colors = {k:v['color'] for k,v in args.plot_info.items()}
    #hatches = {k:v['hatch'] for k,v in args.plot_info.items()}
    colors = args.colors
    hatches = None

    for name, analysis in analyses_together.items():
        tmp = data.copy()
        sdct = {'res_type': res_types,
                'dataset': args.chans,
                'analysis':analysis}
        tmp = select_from_dataframe(tmp, conditions_dict=sdct)

        #make bland altman style plot with boxplots for diff


def old_barplot_performance_over_timepoints(
    data: pd.DataFrame,
    outcomes: List[str],
    timepoint_col: str,
    model_label: str = "experiment",     # with vs without augmentation
    res_type: str = "res_type",          # Artery / Vein
    height: float = 3.5,
    aspect: float = 1.4,
    row_order: Optional[List[str]] = None,
    col_order: Optional[List[str]] = None,
    time_order: Optional[List] = None,
    hue_order: Optional[List[str]] = None,
    sharey: bool = False,               # IMPORTANT: keep row-specific scales
    title: Optional[str] = None,
    palette: str | dict = "tab10",
    legend: bool = True,
    rotate_xticks: int = None,
):
    """
    Rows   → outcomes (metrics)
    Cols   → res_type (Artery / Vein)
    X-axis → timepoint_col
    Hue    → model_label (e.g., with vs without contrast-phase augmentation)
    Bars   → mean; errorbar → SE
    """

    df = data.copy()

    # -----------------------------
    # reshape outcomes → rows
    # -----------------------------
    id_vars = [model_label, res_type, timepoint_col]
    df = df.melt(
        id_vars=id_vars,
        value_vars=outcomes,
        var_name="Outcome",
        value_name="Value"
    )

    # -----------------------------
    # ordering
    # -----------------------------
    if row_order is not None:
        df["Outcome"] = pd.Categorical(df["Outcome"], categories=row_order, ordered=True)
    else:
        row_order = df["Outcome"].unique().tolist()

    if col_order is not None:
        df[res_type] = pd.Categorical(df[res_type], categories=col_order, ordered=True)
    else:
        col_order = df[res_type].unique().tolist()

    if time_order is not None:
        df[timepoint_col] = pd.Categorical(df[timepoint_col], categories=time_order, ordered=True)

    if hue_order is not None:
        df[model_label] = pd.Categorical(df[model_label], categories=hue_order, ordered=True)

    # -----------------------------
    # palette
    # -----------------------------
    if isinstance(palette, str):
        palette = dict(
            zip(
                df[model_label].astype(str).unique(),
                sns.color_palette(palette, df[model_label].nunique())
            )
        )

    # -----------------------------
    # main plot (bars with SE)
    # -----------------------------
    g = sns.catplot(
        data=df,
        x=timepoint_col,
        y="Value",
        hue=model_label,
        col=res_type,
        row="Outcome",
        kind="bar",
        errorbar="se",            # vertical SE bars (what you want)
        height=height,
        aspect=aspect,
        palette=palette,
        sharey=False,            # keep independent y per facet
        legend=legend,
    )

    # -----------------------------
    # aesthetics
    # -----------------------------
    # clean column titles
    for j, col_name in enumerate(g.col_names):
        for i in range(len(g.row_names)):
            g.axes[i][j].set_title(str(col_name))

    # y-labels = metric names on first column only
    for i, outcome in enumerate(g.row_names):
        g.axes[i][0].set_ylabel(outcome)

    for j in range(len(g.col_names)):
        g.axes[-1][j].set_xlabel(timepoint_col)

    # x-axis labels and ticks on EVERY row
    for ax in g.axes.flat:
        ax.tick_params(axis="x", bottom=True, labelbottom=True)
        if rotate_xticks is not None:
            for label in ax.get_xticklabels():
                label.set_rotation(rotate_xticks)
                label.set_ha("right")

    # grid + tick rotation
    for ax in g.axes.flat:
        ax.grid(axis="y", alpha=0.3)
        for label in ax.get_xticklabels():
            if rotate_xticks is not None:
                label.set_rotation(rotate_xticks)
                label.set_ha("right")

    # -----------------------------
    # row-wise y-limits (shared across columns)
    # -----------------------------
    row_ylims = {}
    for outcome in df["Outcome"].unique():
        vals = df.loc[df["Outcome"] == outcome, "Value"].dropna()
        if len(vals) == 0:
            continue

        ymin = vals.min()
        ymax = vals.max()

        pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.05 * abs(ymax)
        row_ylims[outcome] = (ymin - pad, ymax + pad)

    # apply shared y-limits per row
    for i, outcome in enumerate(g.row_names):
        if outcome not in row_ylims:
            continue
        ymin, ymax = row_ylims[outcome]
        for j in range(len(g.col_names)):
            g.axes[i][j].set_ylim(ymin, ymax)

    if title is not None:
        g.figure.suptitle(title, y=1.02)

    return g

def barplot_performance_over_timepoints(
    data: pd.DataFrame,
    outcomes: list[str],
    timepoint_col: str,
    model_label: str = "experiment",
    res_type: str = "res_type",
    height: float = 3.5,
    aspect: float = 1.4,
    row_order: list[str] | None = None,
    col_order: list[str] | None = None,
    time_order: list | None = None,
    hue_order: list[str] | None = None,
    title: str | None = None,
    palette: str | dict = "tab10",
    legend: bool = True,
    rotate_xticks: int | None = None,
):
    """
    Rows   → outcomes (metrics)
    Cols   → res_type (e.g., Artery / Vein)
    X-axis → timepoint_col (shown on every row)
    Hue    → model_label
    Bars   → mean; errorbar → SE
    Y-axis → shared per row (not starting at 0)
    """
    df = data.copy()

    # -----------------------------
    # reshape outcomes → rows
    # -----------------------------
    id_vars = [model_label, res_type, timepoint_col]
    df = df.melt(
        id_vars=id_vars,
        value_vars=outcomes,
        var_name="Outcome",
        value_name="Value",
    )

    # -----------------------------
    # ordering
    # -----------------------------
    if row_order is not None:
        df["Outcome"] = pd.Categorical(df["Outcome"], categories=row_order, ordered=True)
    else:
        row_order = df["Outcome"].unique().tolist()

    if col_order is not None:
        df[res_type] = pd.Categorical(df[res_type], categories=col_order, ordered=True)
    else:
        col_order = df[res_type].unique().tolist()

    if time_order is not None:
        df[timepoint_col] = pd.Categorical(
            df[timepoint_col], categories=time_order, ordered=True
        )

    if hue_order is not None:
        df[model_label] = pd.Categorical(
            df[model_label], categories=hue_order, ordered=True
        )

    # -----------------------------
    # palette
    # -----------------------------
    if isinstance(palette, str):
        palette = dict(
            zip(
                df[model_label].astype(str).unique(),
                sns.color_palette(palette, df[model_label].nunique()),
            )
        )

    # -----------------------------
    # compute row-wise y-limits
    # -----------------------------
    row_ylims = {}
    for outcome in row_order:
        vals = df.loc[df["Outcome"] == outcome, "Value"].dropna()
        if len(vals) == 0:
            continue
        ymin, ymax = vals.min(), vals.max()
        pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.05 * abs(ymax)
        row_ylims[outcome] = (ymin - pad, ymax + pad)

    # -----------------------------
    # main plot
    # -----------------------------
    g = sns.catplot(
        data=df,
        x=timepoint_col,
        y="Value",
        hue=model_label,
        col=res_type,
        row="Outcome",
        kind="bar",
        errorbar="se",
        height=height,
        aspect=aspect,
        palette=palette,
        sharey=False,   # we handle row-sharing manually
        legend=legend,
    )

    # -----------------------------
    # titles and labels
    # -----------------------------
    for j, col_name in enumerate(g.col_names):
        for i in range(len(g.row_names)):
            g.axes[i][j].set_title(str(col_name))

    for i, outcome in enumerate(g.row_names):
        g.axes[i][0].set_ylabel(outcome)

    # -----------------------------
    # x-axis on every row
    # -----------------------------
    for ax in g.axes.flat:
        ax.tick_params(axis="x", bottom=True, labelbottom=True)
        if rotate_xticks is not None:
            for label in ax.get_xticklabels():
                label.set_rotation(rotate_xticks)
                label.set_ha("right")
        ax.grid(axis="y", alpha=0.3)

    # -----------------------------
    # apply shared y-limits per row
    # -----------------------------
    for i, outcome in enumerate(g.row_names):
        if outcome not in row_ylims:
            continue
        ymin, ymax = row_ylims[outcome]
        for j in range(len(g.col_names)):
            g.axes[i][j].set_ylim(ymin, ymax)

    if title is not None:
        g.figure.suptitle(title, y=1.02)

    return g

def simcta_barplot(data):

    analyses_together = {'minCC0.01mL':['headmasks_minCC0.01mL', 'headmasks_adj509_dil5_minCC0.01mL'],
                         'minCC0mL': ['headmasks_minCC1e-07mL', 'headmasks_adj509_dil5_minCC1e-07mL'],
                         }

    #colors = {k:v['color'] for k,v in args.plot_info.items()}
    #hatches = {k:v['hatch'] for k,v in args.plot_info.items()}
    colors = args.colors
    hatches = None

    for name, analysis in analyses_together.items():
        tmp = data.copy()
        sdct = {'res_type': res_types,
                'dataset': args.chans,
                'analysis':analysis,
                'experiment': ['nnUNet-org w/o', 'nnUNet-org w']
                }
        tmp = select_from_dataframe(tmp, conditions_dict=sdct)
        tmp['Time to peak arterial phase (seconds)'] = tmp['channel'].astype(int)

        time_order = tmp['Time to peak arterial phase (seconds)'].unique()
        time_order.sort()

        barplot_performance_over_timepoints(
            data=tmp[tmp['res_type'].isin(['Artery', 'Vein'])],
            outcomes=outcomes,
            timepoint_col='Time to peak arterial phase (seconds)',  # time to peak arterial
            model_label="experiment",  # with vs without augmentation
            res_type="res_type",  # Artery / Vein
            col_order=["Artery", "Vein"],
            time_order=time_order,
            title="Effect of contrast-phase augmentation over time",
            hue_order=['nnUNet-org w/o', 'nnUNet-org w'],
            palette=colors
        )
        plt.show()

        print(1)


def cs_wide_to_long(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    sep: str = "--",
):
    if cols is None:
        cols = []

    # reset index to keep row identity
    df = df.reset_index()

    # composite metric columns
    value_cols = [c for c in df.columns if sep in c]

    # columns to keep as identifiers

    # melt composite columns
    long = df.melt(
        id_vars=cols,
        value_vars=value_cols,
        var_name="hue_metric",
        value_name="value",
    )

    # split hue / metric
    long[["hue", "metric"]] = long["hue_metric"].str.split(
        sep, n=1, expand=True
    )

    # pivot metrics back to columns
    out = (
        long
        .pivot(
            index=cols + ["hue"],
            columns="metric",
            values="value",
        )
        .reset_index()
    )

    # clean column names
    out.columns.name = None

    return out


def fetch_qcs_data(args, cdata):

    dir_qcs = os.path.join(args.cta_pred, 'collateral_scores')
    qcs_files = {k:os.path.join(dir_qcs, k+'_collateral_scores.xlsx') for k in args.experiments.keys()}

    ih_left = cdata['Infarct_side_left'].to_dict()

    for k, v in tqdm(qcs_files.items()):
        aqcs_col, vqcs_col = f'{k}--aQCS', f'{k}--vQCS'
        if aqcs_col in cdata.columns and vqcs_col in cdata.columns:
            continue
        if not os.path.exists(v):
            continue

        qcs = pd.read_excel(v, index_col='ID')
        a, v, both = qcs[qcs['mask_val']=='artery'], qcs[qcs['mask_val']=='vein'], qcs[qcs['mask_val']=='av_both']
        #if infarct left 2/1=left/right hemisphere is the correct qcs
        aqcs, vqcs, a_both, v_both = {}, {}, {}, {}
        for ID in qcs.index:
            if ID in ih_left:
                if ih_left[ID]==1:
                    colscore = '_2/1'
                elif ih_left[ID]==0:
                    colscore = '_1/2'
                else:
                    colscore = '_auto_score'
            else:
                colscore = '_auto_score'
            aqcs[ID] = a[f'{args.artery_score}{colscore}'].loc[ID]
            vqcs[ID] = v[f'{args.vein_score}{colscore}'].loc[ID]
            a_both[ID] = both[f'{args.artery_score}{colscore}'].loc[ID]
            v_both[ID] = both[f'{args.vein_score}{colscore}'].loc[ID]
        aqcs = pd.DataFrame.from_dict(aqcs, orient='index', columns=[f'{k}--aQCS'])
        vqcs = pd.DataFrame.from_dict(vqcs, orient='index', columns=[f'{k}--vQCS'])
        a_both = pd.DataFrame.from_dict(a_both, orient='index', columns=[f'{k}--aQCSboth'])
        v_both = pd.DataFrame.from_dict(v_both, orient='index', columns=[f'{k}--vQCSboth'])
        cdata = cdata.merge(aqcs, left_index=True, right_index=True, how='left')
        cdata = cdata.merge(vqcs, left_index=True, right_index=True, how='left')
        cdata = cdata.merge(a_both, left_index=True, right_index=True, how='left')
        cdata = cdata.merge(v_both, left_index=True, right_index=True, how='left')

    #rename collateral scores for plotting
    tan_dct = {0:'0%', 1:'1–50%', 2:'51–99%', 3:'100%'}
    tan_dct = {0:'0', 1:'1', 2:'2', 3:'3'}
    cdata['Tan collateral score'] = pd.Categorical(
                                            cdata["CTA_Tan_collateral_score"].map(tan_dct),
                                            categories=tan_dct.values(),
                                            ordered=True
                                        )

    cdata['Cortical vein opacification score'] =  pd.Categorical(
                                                            pd.cut(
                                                                cdata["COVES"],
                                                                bins=[-0.1, 2, 4, 6],
                                                                labels=["0–2", "3–4", "5–6"],
                                                                right=True
                                                            ),
                                                            categories=["0–2", "3–4", "5–6"],
                                                            ordered=True
                                                        )

    return cdata

def load_clinical_data(args):

    cdata = pd.read_excel(args.clinical_data, index_col='ID')
    #cdata = fetch_qcs_data(args, cdata)
    return cdata



def plot_tan_coves(
    data,
    aqcs_col,
    vqcs_col,
    hue,
    tan_col="Tan collateral score",
    coves_col="COVES_group",
    figsize=(8, 4),
    art_color='#E74C3C',
    vein_color='#3498DB'
):

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    art_kwargs = dict(
        data=data,
        y=aqcs_col,
        hue=hue,
        color=art_color,
        showfliers=False,
    )

    vein_kwargs = dict(
        data=data,
        y=vqcs_col,
        hue=hue,
        color=vein_color,
        showfliers=False,
    )

    # ---- Panel A: Tan ----
    sns.boxplot(
        x=tan_col,
        ax=axes[0],
        **art_kwargs,
    )

    axes[0].set_xlabel(tan_col)
    axes[0].set_ylabel(aqcs_col)
    axes[0].tick_params(axis="y")

    axes[0].text(
        0.02, 0.95, "A",
        transform=axes[0].transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    # ---- Panel B: COVES ----
    sns.boxplot(
        x=coves_col,
        ax=axes[1],
        legend=False,
        **vein_kwargs,
    )

    axes[1].set_xlabel(coves_col)
    axes[1].set_ylabel(vqcs_col)
    axes[1].tick_params(axis="y")

    axes[1].text(
        0.02, 0.95, "B",
        transform=axes[1].transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )



    plt.tight_layout()

    return fig, axes


def plot_av_outcome(
    data,
    y_var: str,
    a: str,
    v: str,
    figsize=(16, 6)
):
    """
    Plot artery, vein, and their mismatch against a continuous y variable.

    Args:
        data (pd.DataFrame): Data with columns y_var, a, v
        y_var (str): Name of continuous outcome variable
        a (str): Name of artery variable
        v (str): Name of vein variable
        figsize (tuple): Figure size
        scatter_kwargs (dict): Optional kwargs passed to plt.scatter

    Returns:
        fig, axes
    """

    for col in [y_var, a, v]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not in dataframe")

    # --- compute mismatch ---
    mismatch = data[a] - (data[[a, v]].mean(axis=1))
    data = data.copy()
    data["aQCS - vQCS mismatch"] = mismatch

    # --- prepare plotting ---
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=False)
    fig.tight_layout(pad=4.0)

    # Panel titles
    x_vars = [a, v, "aQCS - vQCS mismatch"]

    for ax, x in zip(axes, x_vars):
        sns.regplot(
            ax=ax,
            x=data[x],
            y=data[y_var],
            scatter_kws={'alpha': 0.6, 'color':'gray'},
            line_kws={'lw': 1.5, 'color':'black'}
        )
        ax.set_xlabel(x)
        ax.set_ylabel(y_var)
        ax.grid(True)

        if x == a:
            # ** lower left annotation with arrow **
            ax.text(
                0.02, 0.005,
                "→ more arterial inflow",
                transform=ax.get_xaxis_transform(),
                fontsize=10,
                va="bottom"
            )

            ax.text(
                0.02, 0.98, "A",
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                va="top",
            )
        elif x == v:
            # ** lower left annotation with arrow **
            ax.text(
                0.02, 0.005,
                "→ more venous outflow",
                transform=ax.get_xaxis_transform(),
                fontsize=10,
                va="bottom"
            )

            ax.text(
                0.02, 0.98, "B",
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                va="top",
            )

        elif x == "aQCS - vQCS mismatch":
            ax.axvline(0, linestyle="--", color="gray")

            # text left of x=0 line (data x = 0)
            ax.text(
                -0.01, 0.005,
                "more inflow than outflow ←",
                transform=ax.get_xaxis_transform(),
                fontsize=9,
                ha="right",
                va="bottom"
            )

            # text right of x=0 line
            ax.text(
                0.01, 0.005,
                "→ more outflow than inflow",
                transform=ax.get_xaxis_transform(),
                fontsize=10,
                ha="left",
                va="bottom"
            )

            ax.text(
                0.02, 0.98, "C",
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                va="top",
            )

        ymin, ymax = ax.get_ylim()
        pad = 0.03 * (ymax - ymin)  # 5% of the current range
        ax.set_ylim(min(ymin, -pad), ymax)

    plt.tight_layout()

    return fig, axes

def multiple_spearmans(data, xs, yx, nan_policy="omit"):
    out = []
    for x,y in zip(xs, yx):
        rho, p = stats.spearmanr(data[x].astype(float), data[y].astype(float), nan_policy=nan_policy)
        out.extend([rho,p])
    return out


def tan_coves_plot(input_data, dir_fig, qcs_exp, tan_var=None, coves_var=None, subgroups=[]):

    if dir_fig is not None:
        os.makedirs(dir_fig, exist_ok=True)

    input_data = fetch_qcs_data(args, input_data)
    input_data['Post transfer DWI infarct volume (log₁₀ mL)'] = np.log10(np.clip(input_data['blt_vol'].astype(float), 0.99,1000))
    yvar = 'Post transfer DWI infarct volume (log₁₀ mL)'
    input_data[yvar] = input_data[yvar].replace([np.inf, -np.inf], np.nan)
    data = cs_wide_to_long(input_data, cols=['ID',
                                       tan_var,
                                       coves_var,
                                       *subgroups,
                                       "CTA_Tan_collateral_score",
                                       'COVES',
                                       yvar,
                                       ], sep="--")

    out = [] #for correlations
    for a,v in qcs_exp:
        data[a] = np.clip(data[a],0.0, 1.0)
        data[v] = np.clip(data[v],0.0, 1.0)
        for exp in data['hue'].unique():
            tmp = data[data['hue']==exp]

            sns.scatterplot(tmp, x=a, y=v)
            plt.savefig(os.path.join(dir_fig, f'{exp}_{a}_vs_{v}_scatter.png'))
            plt.title(args.experiments[exp])
            plt.show()

            plot_tan_coves(
                tmp,
                aqcs_col=a,
                vqcs_col=v,
                hue=None,
                tan_col=tan_var,
                coves_col=coves_var,
                figsize=(8, 4),
            )
            plt.savefig(os.path.join(dir_fig, f'{exp}_{a}_and_{v}_boxplot.png'))
            plt.show()

            plot_av_outcome(tmp, yvar, a, v)
            plt.savefig(os.path.join(dir_fig, f'{exp}_{a}_and_{v}_vs_DWI_scatter.png'))
            plt.show()

            tmp.loc[:,"aQCS - vQCS mismatch"]  = tmp[a] - (tmp[[a, v]].mean(axis=1))

            sprmn = multiple_spearmans(tmp,
                                   [a, tan_var, 'COVES', tan_var, 'COVES', a, v, "aQCS - vQCS mismatch"],
                                   [v, a, v, yvar, yvar, yvar, yvar, yvar])

            out.append([exp, 'main', a,v, *sprmn])

            for subgroup in subgroups:
                subdir_fig = os.path.join(dir_fig, subgroup)
                os.makedirs(subdir_fig, exist_ok=True)

                tmp = data[(data['hue']==exp) & data[subgroup]==True]
                sns.scatterplot(tmp, x=a, y=v)
                plt.savefig(os.path.join(subdir_fig, f'{exp}_{a}_vs_{v}_{subgroup}True_scatter.png'))
                plt.show()

                plot_tan_coves(
                    tmp,
                    aqcs_col=a,
                    vqcs_col=v,
                    hue=None,
                    tan_col=tan_var,
                    coves_col=coves_var,
                    figsize=(8, 4),
                )
                plt.savefig(os.path.join(subdir_fig, f'{exp}_{a}_and_{v}_{subgroup}True_boxplot.png'))
                plt.show()

                tmp["aQCS - vQCS mismatch"] = tmp[a] - (tmp[[a, v]].mean(axis=1))
                sprmn = multiple_spearmans(tmp,
                                           [a, tan_var, 'COVES', tan_var, 'COVES', a, v, "aQCS - vQCS mismatch"],
                                           [v, a, v, yvar, yvar, yvar, yvar, yvar])

                out.append([exp,  f'{subgroup}True', a, v, *sprmn])

                tmp = data[(data['hue'] == exp) & data[subgroup] == False]
                sns.scatterplot(tmp, x=a, y=v)
                plt.savefig(os.path.join(subdir_fig, f'{exp}_{a}_vs_{v}_{subgroup}False_boxplot.png'))
                plt.show()

                plot_tan_coves(
                    tmp,
                    aqcs_col=a,
                    vqcs_col=v,
                    hue=None,
                    tan_col=tan_var,
                    coves_col=coves_var,
                    figsize=(8, 4),
                )
                plt.savefig(os.path.join(subdir_fig, f'{exp}_{a}_and_{v}_{subgroup}False_boxplot.png'))
                plt.show()

                sprmn = multiple_spearmans(tmp,
                                           [a, tan_var, 'COVES', tan_var, 'COVES', a, v, "aQCS - vQCS mismatch"],
                                           [v, a, v, yvar, yvar, yvar, yvar, yvar])

                out.append([exp,  f'{subgroup}False', a, v, *sprmn])

    out = pd.DataFrame(out, columns=['experiment', 'subgroup', 'aqcs_col', 'vqcs_col',
                                        'rho_aqcs_vqcs', 'p_aqcs_vqcs',
                                        'rho_tan_aqcs', 'p_tan_aqcs',
                                        'rho_coves_vqcs', 'p_coves_vqcs',
                                        'rho_tan_dwi', 'p_tan_dwi',
                                        'rho_coves_dwi', 'p_coves_dwi',
                                        'rho_aqcs_dwi', 'p_aqcs_dwi',
                                        'rho_vqcs_dwi', 'p_vqcs_dwi',
                                        'rho_mismatch_dwi', 'p_mismatch_dwi'
                                     ])
    out.to_excel(os.path.join(dir_fig, 'tan_coves_correlations.xlsx'), index=False)
    return out


if __name__ == "__main__":
    args = init_args()
    args = update_args_with_yaml(args, load_yaml_config(args.yml_args))
    dir_figures = os.path.join(args.p_out, 'figures')
    os.makedirs(dir_figures, exist_ok=True)
    f_perf_table = os.path.join(args.p_out, 'performance_summary.xlsx')
    f_diff_table = os.path.join(args.p_out, 'differences_stats.xlsx')
    data = get_test_results(args.p_out, overwrite=args.overwrite)
    data = data_prep(data)

    exp_order = list([exp.split(' ')[0] for exp in args.experiments.values() if 'w/o' in exp])  # extract base names without ' w' or ' wo'
    exp_combis = list((exp.split(' ')[0] + ' w', exp) for exp in args.experiments.values() if 'w/o' in exp)
    if not os.path.exists(f_diff_table) or args.overwrite:
        diff = compute_id_wise_differences(data, round_dct=args.round_dct, exp_combis=exp_combis)
        diff = channel_cols(diff, select_chans=getattr(args, 'chans', None))
        #add simCTA dataset splits again
        single_stat, time_stat = stat_compare_differences(diff, round_dct=args.round_dct, exp_col= 'experiment_diff')
        write_multitab_excel({'single':single_stat, 'time':time_stat, 'diff':diff}, f_diff_table)
    else:
        single_stat = pd.read_excel(f_diff_table, sheet_name='single')
        time_stat = pd.read_excel(f_diff_table, sheet_name='time')
        diff = pd.read_excel(f_diff_table, sheet_name='diff')

    #make summary performance tables
    if not os.path.exists(f_perf_table) or args.overwrite:
        summary_table = mean_sd_table(data,
                                        partition_by=['main_dataset','analysis','dataset', 'experiment', 'res_type'],
                                        rounding = args.round_dct if hasattr(args, 'round_dct') else None,
                                        use_plus_minus = True,
                                        use_se=True
                                        )
        summary_table.to_excel(f_perf_table)
        #TODO: make separate tab with final results
        m = build_metric_tables(summary_table, metrics=args.round_dct.keys(), exp_order=exp_order)
        write_multitab_excel(m, f_perf_table.replace('.xlsx', '_per_metric.xlsx'))
    else:
        summary_table = pd.read_excel(f_perf_table)


    if hasattr(args, 'chans'):
        chan_dct = {chan: int(chan[1:])*-1 if 'm' in chan else int(chan[1:]) for chan in args.chans}

    outcomes = ['Dice', 'clDice', 'AVD'] #, 'AHD','HD95',
    res_types = ['Artery', 'Vein', 'Any Vessel']
    exp = []
    subplot_id = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

    #Tan and Coves comparison
    cdata = load_clinical_data(args)
    # subgroup <0.7mm vs >0.7mm (exclude all >1.5mm)
    cdata['thin_subgroup'] = cdata['z_spacing']<=0.7
    #make plot 2 models aQCS vs Tan and vQCS vs COVES
    # tan_coves_plot(cdata,
    #                os.path.join(dir_figures, 'tan_coves'),
    #                qcs_exp=[('aQCS', 'vQCS'), ('aQCSboth', 'vQCSboth')],
    #                # subgroups=['thin_subgroup'],
    #                tan_var='Tan collateral score',
    #                coves_var='Cortical vein opacification score')

    simcta_barplot(data)
    #barplots()
    #simCTA results
    # simcta_plots(data,
    #              [exp_combis[0]],
    #              #chan_dct,
    #              {k:v for k,v in chan_dct.items() if abs(v)<=6},
    #              time_stat,
    #              outcomes=['Dice', 'clDice', 'AVD'],
    #              res_types=['Artery', 'Vein'],
    #              dir_figures=os.path.join(dir_figures,'simcta'))


    print(1)

    #boxplots for good and poor
    # p = 'other/SU_CTP_todo/pertime_gt_headmasks_adj509_dil0_minCC0.01mL'
    # pnew = p+'_BLUE'
    # os.makedirs(pnew, exist_ok=True)
    # for f in os.listdir(p):
    #     if 'SU0' in f and f.endswith('.nii.gz'):
    #         mask = sitk.ReadImage(os.path.join(p, f))
    #         arr = sitk.GetArrayFromImage(mask)
    #         arr[arr==2]=4
    #         sitk.WriteImage(np2sitk(arr,mask), os.path.join(pnew, f))
