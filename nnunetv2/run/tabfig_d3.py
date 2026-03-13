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
from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
from scipy.stats import ttest_1samp
from nnunetv2.my_utils.utils import init_args, update_args_with_yaml, load_yaml_config, \
    get_nnUNet_paths, get_experiments, NiftiLoader, get_path_dict, combine_excel_files, np2sitk, write_multitab_excel
from nnunetv2.run.multichan_multidata_tabfig import multiple_cat_cont_comps, plot_av_outcome
from jonckheere_test import jonckheere_test


def plot_Maas(
    data,
    aqcs_col,
    vqcs_col,
    maas_col="Maas collateral score",
    hue=None,
    figsize=(8,4),
    art_color="#E74C3C",
    vein_color="#3498DB",
    violin=False,
    subplot_labels=None,
    ylim=None,
    art_kwargs=None,
    vein_kwargs=None,
):
    if subplot_labels is None:
        subplot_labels = []

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ensure Maas ordered
    if not str(data[maas_col].dtype).startswith("category"):
        data = data.copy()
        data[maas_col] = data[maas_col].astype("category")
        data[maas_col] = data[maas_col].cat.set_categories(
            sorted(data[maas_col].dropna().unique()),
            ordered=True
        )

    # default kwargs
    if art_kwargs is None:
        art_kwargs = dict(
            data=data,
            x=maas_col,
            y=aqcs_col,
            hue=hue,
            color=art_color,
        )
        if not violin:
            art_kwargs["showfliers"] = False

    if vein_kwargs is None:
        vein_kwargs = dict(
            data=data,
            x=maas_col,
            y=vqcs_col,
            hue=hue,
            color=vein_color,
        )
        if not violin:
            vein_kwargs["showfliers"] = False

    # ---- Panel A: arterial quantitative score ----
    if violin:
        sns.violinplot(ax=axes[0], **art_kwargs)
    else:
        sns.boxplot(ax=axes[0], **art_kwargs)

    axes[0].set_xlabel("Maas collateral score")
    if aqcs_col!='aQCS':
        axes[0].set_ylabel('aQCS {}'.format(aqcs_col.split("--")[1].replace('_', ' ')))
    else:
        axes[0].set_ylabel(aqcs_col)
    axes[0].set_title("Arterial quantitative collateral score")

    if ylim is not None:
        axes[0].set_ylim(ylim)

    if subplot_labels:
        axes[0].text(
            0.02, 0.95, subplot_labels[0],
            transform=axes[0].transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

    # ---- Panel B: venous quantitative score ----
    if violin:
        sns.violinplot(ax=axes[1], **vein_kwargs)
    else:
        sns.boxplot(ax=axes[1], **vein_kwargs)

    axes[1].set_xlabel("Maas collateral score")
    if vqcs_col!='vQCS':
        axes[1].set_ylabel('vQCS {}'.format(vqcs_col.split("--")[1].replace('_', ' ')))
    else:
        axes[1].set_ylabel(vqcs_col)
    axes[1].set_title("Venous quantitative collateral score")

    if ylim is not None:
        axes[1].set_ylim(ylim)

    if subplot_labels:
        axes[1].text(
            0.02, 0.95, subplot_labels[1],
            transform=axes[1].transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

    plt.tight_layout()

    return fig, axes


def plot_qcs_dists(data, cols, dir_sav):
    os.makedirs(dir_sav, exist_ok=True)

    for c in cols:
        plt.figure(figsize=(10,6))
        sns.histplot(data[c], kde=True)
        plt.title(f'Distribution of {c}')
        plt.xlabel(c)
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(dir_sav, f'{c}_distribution.png'))
        plt.close()

def get_data(args, maas_score = 'Maas__1_5_', fiv='infarctVol_24h'):

    cdata = pd.read_excel(args.f_cdata)
    tabres = pd.read_excel(args.f_tabres)
    #bin maas score in 1-2, 3, 4-5
    print(cdata[maas_score].value_counts().sort_index())
    cdata['Maas score'] = pd.cut(cdata[maas_score], bins=[0, 2, 3, 5], labels=['1-2', '3', '4-5'])

    cdata['24h volume (ml)'] = cdata[fiv]
    cdata['24h volume (log10 ml)'] = np.log10(cdata[fiv])
    #'manual_exclude4tanv2--volume--artery','manual_exclude4tanv2--volume--vein'
    cdata['aQCS'] = cdata['manual_exclude4tanv2--volume--artery']
    cdata['vQCS'] = cdata['manual_exclude4tanv2--volume--vein']

    return cdata, tabres

def maas_qcs_comp(data, maas, aqcs_col, vqcs_col, nan_policy='omit'):

    if nan_policy == "omit":
        mask = data[[maas, aqcs_col, vqcs_col]].notnull().all(axis=1)
        tmp = data.copy()[mask]
    else:
        tmp = data.copy()

    rho1, p1 = stats.spearmanr(tmp[maas].astype(float), tmp[aqcs_col].astype(float), nan_policy=nan_policy)
    jt1 = jonckheere_test(tmp[maas].astype(float), tmp[aqcs_col].astype(float))

    rho2, p2 = stats.spearmanr(tmp[maas].astype(float), tmp[vqcs_col].astype(float), nan_policy=nan_policy)
    jt2 = jonckheere_test(tmp[maas].astype(float), tmp[vqcs_col].astype(float))

    dct = {
        aqcs_col: {
            'spearman_rho': rho1,
            'spearman_p': p1,
            'jonchheere_statistic': jt1.statistic,
            'jonckheere_p': jt1.p_value

        },
        vqcs_col: {
            'spearman_rho': rho2,
            'spearman_p': p2,
            'jonckheere_statistic': jt2.statistic,
            'jonckheere_p': jt2.p_value
        }
    }
    out = pd.DataFrame(dct).T

    return out


if __name__ == "__main__":
    args = init_args()
    args = update_args_with_yaml(args, load_yaml_config(args.yml_args))
    org_maas = 'Maas__1_5_'

    cdata, tabres = get_data(args, maas_score=org_maas)

    p_res = os.path.join(args.p_out, 'test_results')
    #1) plot all distributions
    #plot_qcs_dists(cdata, [c for c in cdata if 'manual_' in c], os.path.join(p_res, 'qcs_dists'))

    coll_pairs_for_plot = {
        'baseline_vol': ['manual_exclude4tanv2--volume--artery', 'manual_include4coves--volume--vein'],
        'baseline_dens': ['manual_exclude4tanv2--median_density--artery', 'manual_include4coves--median_density--vein'],
        'tanv2_vol': ['aQCS','vQCS'],
        'tanv2_dens': ['manual_exclude4tanv2--median_density--artery','manual_exclude4tanv2--median_density--vein'],
        }

    #2) plot all vs Maas
    dir_maas = os.path.join(p_res, 'maas_plots')
    os.makedirs(dir_maas, exist_ok=True)
    dir_fiv = os.path.join(p_res, 'fiv')
    os.makedirs(dir_fiv, exist_ok=True)
    out = []
    for name, [aqcs_col, vqcs_col] in coll_pairs_for_plot.items():
        plot_av_outcome(cdata, '24h volume (ml)', aqcs_col, vqcs_col, add_direction_txt=False)
        plt.savefig(os.path.join(dir_fiv, f'24h_vol_{name}.png'))
        plt.show()

        plot_av_outcome(cdata, '24h volume (log10 ml)', aqcs_col, vqcs_col, add_direction_txt=False)
        plt.savefig(os.path.join(dir_fiv, f'log10_24h_vol_{name}.png'))
        plt.show()

        plot_Maas(cdata, aqcs_col,vqcs_col, maas_col="Maas score")
        plt.savefig(os.path.join(dir_maas, f'Maas_{name}.png'))
        plt.show()
        #stat comparison with jt and correlation
        tmp = maas_qcs_comp(cdata, org_maas, aqcs_col, vqcs_col, nan_policy='omit')
        out.append(tmp)

    out = pd.concat(out)
    out.to_excel(os.path.join(dir_maas , 'maas_qcs_stats.xlsx'))

    #3) compar qcs with 24h volumes



    print(1)