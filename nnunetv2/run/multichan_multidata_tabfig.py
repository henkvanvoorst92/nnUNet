import os
import itertools
import pandas as pd
import ast
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
from nnunetv2.my_utils.utils import np2sitk, image_or_path_load, sitk_dilate_mm

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

if __name__ == "__main__":
    args = init_args()
    args = update_args_with_yaml(args, load_yaml_config(args.yml_args))

    f_perf_table = os.path.join(args.p_out, 'performance_summary.xlsx')

    data = get_test_results(args.p_out, overwrite=args.overwrite)
    data['FPR'] *= 1000 #show FPR per 1000 as it is very small
    data['res_type'] =data['res_type'].replace({'macro_avg':'Macro-average', 'micro-avg':'Micro-average', 1:'Artery', 2:'Vein', 0:'Any vessel'})
    data['experiment'] = data['experiment'].replace(args.experiments)
    exp_order = list([exp.split(' ')[0] for exp in args.experiments.values() if 'w/o' in exp])  # extract base names without ' w' or ' wo'

    #make summary performance tables
    if not os.path.exists(f_perf_table) or args.overwrite:
        summary_table = mean_sd_table(data,
                                        partition_by=['dataset', 'experiment', 'res_type'],
                                        rounding = args.round_dct if hasattr(args, 'round_dct') else None,
                                        use_plus_minus = True
                                        )
        #TODO: make separate tab with final results

        summary_table .to_excel(f_perf_table, index=False)
    else:
        summary_table = pd.read_excel(f_perf_table)

    m = build_metric_tables(summary_table, metrics=args.round_dct.keys(), exp_order=exp_order)

    write_multitab_excel(m, f_perf_table.replace('.xlsx', '_per_metric.xlsx'))
    print(1)

    #F1 stanford aross timepoints figure
    #-> repeat

    #f2

    #test_figures(pc, args)
    #stat_res = comparative_stats(pc)

