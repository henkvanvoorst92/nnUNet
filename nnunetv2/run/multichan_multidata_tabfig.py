import os
import itertools
import pandas as pd
import ast
import seaborn as sns
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
            index=["res_type","exp_base"],
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
                index=['ID', 'res_type', 'dataset'],
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
        groups = [ exp_col, 'dataset','res_type']
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
    if multiple_measurements_exp_col in data.columns:
        mm_data = data[data[multiple_measurements_exp_col].notna()]
        mm_groups = [multiple_measurements_exp_col] + groups
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

        if g._legend is not None:
            g._legend.set_title("")
            #g._legend.set_bbox_to_anchor((0.8, 1.1))

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

        fig = g.figure
        fig.tight_layout(pad=1.0, h_pad=1.0, w_pad=1.0)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
        return fig, g

def channel_cols(data, select_chans=None):
    chs, chn, scta = [], [], []
    for ds in data['dataset']:

        chan_str = ds.split('_')[0] if ('_dil' in ds or '_lblCTP' in ds) else None
        if chan_str is not None:
            chan_num = int(chan_str[1:])*-1 if 'm' in chan_str else int(chan_str[1:])
            simcta = ds.split('_')[1]
        else:
            chan_num = chan_str
            simcta = chan_str
        chs.append(chan_str)
        chn.append(chan_num)
        scta.append(simcta)
    data['channel_name'] = chs
    data['channel'] = chn
    data['simCTA'] = scta
    if select_chans is not None:
        data = data[(np.isin(data['channel_name'], select_chans))|(data['channel_name'].isna())]

    return data

def data_prep(data):
    data['FPR'] *= 1000 #show FPR per 1000 as it is very small
    data['res_type'] =data['res_type'].replace({'macro_avg':'Macro-average', 'micro-avg':'Micro-average', 1:'Artery', 2:'Vein', 0:'Any vessel'})
    data['experiment'] = data['experiment'].replace(args.experiments)
    data['model'] = [exp.split(' ')[0] if ' w' in exp else exp for exp in data['experiment']]
    data['model_subtype'] = [exp.split(' ')[1] if ' w' in exp else '' for exp in data['experiment']]
    data['clDice'] = data['cldice']

    data = channel_cols(data, select_chans=getattr(args, 'chans', None))
    return data

def fetch_simcta_stat(stat_table, ediff, ds, outcome, res_type, return_val=None, string_return=''):


    perf_res = stat_table[(stat_table['experiment_diff'] == ediff) &
                           (stat_table['simCTA'] == ds) &
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
    for ds in data['simCTA'].unique():
        if ds is None:
            continue
        for (e1, e2) in exp_combis:
            # cols artery-vein, rows: Dice, clDice, HD95, Betti0
            #two lines for performance (e1 and e2)
            sdct = {'experiment': [e1, e2],
                   'channel': list(chan_dct.values()),
                   'simCTA': ds,
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
    sharey: bool = False,
    title: Optional[str] = None,
    palette: str = "tab10",
):

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


    if title:
        g.figure.suptitle(title, fontsize=14)
        g.figure.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        g.figure.tight_layout()

    return g



if __name__ == "__main__":
    args = init_args()
    args = update_args_with_yaml(args, load_yaml_config(args.yml_args))
    dir_figures = os.path.join(args.p_out, 'figures')
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
        write_multitab_excel({'single':single_stat, 'time':time_stat}, f_diff_table)
    else:
        single_stat = pd.read_excel(f_diff_table, sheet_name='single')
        time_stat = pd.read_excel(f_diff_table, sheet_name='time')

    #make summary performance tables
    if not os.path.exists(f_perf_table) or args.overwrite:
        summary_table = mean_sd_table(data,
                                        partition_by=['dataset', 'experiment', 'res_type'],
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

    outcomes = ['Dice', 'clDice', 'HD95', 'AVD']
    res_types = ['Artery', 'Vein']
    subplot_id = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

    sdct = {'res_type': res_types, 'dataset': ['t0_dil10','cta', 'poorcta']}
    tmp = select_from_dataframe(data, conditions_dict=sdct)
    # # # add stat res in plot for comparison
    # perf_diffs = []
    # for outcome in outcomes:
    #     for res_type in res_types:
    #         plot_str = fetch_simcta_stat(time_stat, ediff, ds, outcome, res_type,
    #                                      return_val='mean±se', string_return='diff: ')
    #         perf_diffs.append(plot_str)
    #
    # n_subplots = len(tmp['res_type'].unique()) * len(outcomes)

    barplot_performance_models(
                                tmp[tmp['res_type']=='Artery'],
                                outcomes,  # rows for performance measures
                                model = "model",  # model type (color) --> main hue
                                model_subtype = "model_subtype",  # lighter shade --> sub hue
                                model_label = 'experiment',
                                res_type = "dataset",#"res_type",  # columns --> artery-vein-any vessel or dataset
                                bar_order = list(args.experiments.values()),
                                palette = args.colors,
                                sharey=False,
                                title = None,
                                )
    plt.show()

    print(1)

    #simCTA results
    simcta_plots(data, exp_combis, chan_dct, time_stat,
                 outcomes=['Dice', 'clDice', 'HD95', 'AVD'],
                 res_types=['Artery', 'Vein'],
                 dir_figures=os.path.join(dir_figures,'simcta'))

    #boxplots for good and poor
