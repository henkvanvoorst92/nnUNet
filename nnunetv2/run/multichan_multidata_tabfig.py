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
from nnunetv2.my_utils.utils import init_args, update_args_with_yaml, load_yaml_config, \
    get_nnUNet_paths, get_experiments, NiftiLoader, get_path_dict, combine_excel_files, np2sitk, write_multitab_excel
from nnunetv2.run.multichan_val import main_processor, main_results_processor
from nnunetv2.my_utils.plots import boxplot_per_class, test_time_plots
from nnunetv2.my_utils.tables import mean_sd_table
from nnunetv2.my_utils.metrics import comparative_stats, compare_multiclass_masks, compare_masks
from nnunetv2.my_utils.utils import np2sitk, image_or_path_load, sitk_dilate_mm, select_from_dataframe

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


def lineplot_multi_outcomes(
        data: pd.DataFrame,
        ys: List[str],  # <--- list of outcome columns
        x: str = 'channel',
        hue: Optional[str] = 'experiment',  # grouping (used for color)
        outcome_hue: bool = True,  # whether to color by outcome as well
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
        panel_text_kwargs: dict = dict(fontsize=16, fontweight='bold', va='top', ha='left'),
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

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
        return fig, ax

    else:
        # faceted by subplot_by
        g = sns.relplot(
            data=plot_data,
            x=x, y='y_val',
            hue=hue_arg,
            kind="line",
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

        g.set_xlabels(xlabel)
        g.set_ylabels(ylabel)

        if g._legend is not None:
            g._legend.set_title(hue_arg)

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

        fig = g.figure
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
        return fig, g

def channel_cols(data):
    chs, chn, scta = [], [], []
    for ds in data['dataset']:

        chan_str = ds.split('_')[0] if ('_dil' in ds or '_lblCTP' in ds) else None
        if chan_str is not None:
            chan_num = int(chan_str[1:])*-1 if 'm' in chan_str else int(chan_str[1:])
            simcta = ds.split('_')[1]
        else:
            chan_num = chan_str
            sim_ta = chan_str
        chs.append(chan_str)
        chn.append(chan_num)
        scta.append(simcta)
    data['channel_name'] = chs
    data['channel'] = chn
    data['simCTA'] = scta
    return data


if __name__ == "__main__":
    args = init_args()
    args = update_args_with_yaml(args, load_yaml_config(args.yml_args))

    f_perf_table = os.path.join(args.p_out, 'performance_summary.xlsx')
    data = get_test_results(args.p_out, overwrite=args.overwrite)
    data['FPR'] *= 1000 #show FPR per 1000 as it is very small
    data['res_type'] =data['res_type'].replace({'macro_avg':'Macro-average', 'micro-avg':'Micro-average', 1:'Artery', 2:'Vein', 0:'Any vessel'})
    data['experiment'] = data['experiment'].replace(args.experiments)
    data['clDice'] = data['cldice']
    data = channel_cols(data)
    exp_order = list([exp.split(' ')[0] for exp in args.experiments.values() if 'w/o' in exp])  # extract base names without ' w' or ' wo'

    #make summary performance tables
    if not os.path.exists(f_perf_table) or args.overwrite:
        summary_table = mean_sd_table(data,
                                        partition_by=['dataset', 'experiment', 'res_type'],
                                        rounding = args.round_dct if hasattr(args, 'round_dct') else None,
                                        use_plus_minus = True
                                        )
        summary_table.to_excel(f_perf_table)
        #TODO: make separate tab with final results
        m = build_metric_tables(summary_table, metrics=args.round_dct.keys(), exp_order=exp_order)
        write_multitab_excel(m, f_perf_table.replace('.xlsx', '_per_metric.xlsx'))
    else:
        summary_table = pd.read_excel(f_perf_table)

    #make figures
    exp_combis = list((exp.split(' ')[0]+' w', exp) for exp in args.experiments.values() if 'w/o' in exp)

    if hasattr(args, 'chans'):
        chan_dct = {chan: int(chan[1:])*-1 if 'm' in chan else int(chan[1:]) for chan in args.chans}
    #simCTA results
    outcomes = ['Dice', 'clDice']
    subplot_id = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    for ds in data['simCTA'].unique():
        if ds is None:
            continue
        for (e1, e2) in exp_combis:
            ds = 'lblCTP'

            sdct = {'experiment': [e1, e2],
                   'channel': list(chan_dct.values()),
                   'simCTA': ds,
                   'res_type':['Artery', 'Vein', 'Any vessel']}
            tmp = select_from_dataframe(data, conditions_dict=sdct)

            lineplot_multi_outcomes(tmp,
                                    ys=['Dice'], x='channel',
                                    hue='experiment', subplot_by = 'res_type',
                                    panel_text = [subplot_id[i] for i in range(len(tmp['res_type'].unique()))]
                                    )
            plt.show()

            lineplot_multi_outcomes(tmp,
                                    ys=['clDice'], x='channel',
                                    hue='experiment', subplot_by = 'res_type',
                                    panel_text = [subplot_id[i] for i in range(len(tmp['res_type'].unique()))]
                                    )
            plt.show()

            print(1)


        #test_figures(pc, args, channel_dct=chan_dct, select_exp=[e1, e2])

        break


    print(1)

    #F1 stanford aross timepoints figure
    #-> repeat

    #f2

    #test_figures(pc, args)
    #stat_res = comparative_stats(pc)

