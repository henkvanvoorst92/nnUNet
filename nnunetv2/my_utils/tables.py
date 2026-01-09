import pandas as pd
import numpy as np

def mean_sd_table(spatial,
                  columns=None,
                  partition_by=None,
                  rounding=None,
                  use_plus_minus=False,
                  use_se=False):
    """
    Summarize metrics with mean and standard deviation, optionally partitioning the data.

    Parameters:
    - spatial (pd.DataFrame): Input DataFrame containing the data.
    - columns (list): List of column names to compute mean and standard deviation for.
    - partition_by (list, optional): List of column names to partition the data by. Default is None.
    - rounding (dict, optional): Dictionary specifying rounding for each column. Default is None.
    - use_plus_minus (bool, optional): If True, format output as "mean ± sd". Default is False.

    Returns:
    - pd.DataFrame: Summary table with mean and standard deviation for each column.
    """
    if partition_by is None:
        partition_by = []

    if columns is None and rounding is not None:
        columns = list(rounding.keys())

    # Group by partition columns if provided
    grouped = spatial.groupby(partition_by) if partition_by else [("", spatial)]

    results = []
    for group_name, group_data in grouped:
        row = {col: group_name[i] if isinstance(group_name, tuple) else group_name for i, col in enumerate(partition_by)}
        row['count_notna'] = group_data[columns[0]].notna().sum()
        row['count'] = len(group_data)
        for col in columns:
            mean = group_data[col].mean()
            std = group_data[col].std()
            se = std / np.sqrt(row['count_notna']) if row['count_notna'] > 0 else np.nan
            if rounding and col in rounding:
                r = rounding[col]
                if r>0:
                    mean = round(mean, r)
                    std = round(std, r)
                    se = round(se, r)
                else:
                    mean = int(round(mean, r)) if not np.isnan(mean) else mean
                    std = int(round(std, r)) if not np.isnan(std) else std
                    se = int(round(se, r)) if not np.isnan(se) else se
            if use_se:
                row[col] = f"{mean} ±{se}" if use_plus_minus else (mean, se)
            else:
                row[col] = f"{mean} ±{std}" if use_plus_minus else (mean, std)
        results.append(row)

    return pd.DataFrame(results)


#
# def spatial_results(spatial,
#                     msm,
#                     metrics,
#                     select_subgroup={},
#                     partition_by=['ih', 'mask','combined','fold'],
#                     round_dct=None):
#
#     if round_dct is None:
#         round_dct = {'Dice':2, 'TPR':2, 'FPR':2, 'PPV':2, 'NPV':2,'AVD':0,'AHD':0}
#
#     pb = [p for p in partition_by if len(spatial[p].unique())>1]
#     res = mean_sd_table(spatial,
#                         columns=metrics,
#                         partition_by=pb,
#                         rounding = round_dct,
#                         use_plus_minus = True
#                         )
#     dfs = {'all':res}
#     for ds in spatial.dataset.unique():
#         dta = spatial[spatial['dataset']==ds]
#         pb = [p for p in partition_by if len(dta[p].unique()) > 1]
#         res = mean_sd_table(dta,
#                             columns=metrics,
#                             partition_by=pb,
#                             rounding = round_dct,
#                             use_plus_minus = True
#                             )
#         dfs[ds] = res
#
#     #subgroup analysis>50mL
#     #select_modality = 'ih--dwi_roi-inclusion_mask--hu0'
#     #select = np.isin(spatial['ID'], msm[(msm['vol']>50)&(msm['exp']==select_modality)]['ID'])
#
#     if len(select_subgroup)>0:
#         for subgroupname, select_IDs in select_subgroup.items():
#             subgroup = spatial[np.isin(spatial['ID'], select_IDs)]
#             pb = [p for p in partition_by if len(subgroup[p].unique()) > 1]
#             res = mean_sd_table(subgroup,
#                                 columns=metrics,
#                                 partition_by=pb,
#                                 rounding = round_dct,
#                                 use_plus_minus = True
#                                 )
#             dfs[f'all{subgroupname}'] = res
#
#             for ds in subgroup.dataset.unique():
#                 dta = subgroup[subgroup['dataset']==ds]
#                 pb = [p for p in partition_by if len(dta[p].unique()) > 1]
#                 res = mean_sd_table(dta,
#                                     columns=metrics,
#                                     partition_by=pb,
#                                     rounding = round_dct,
#                                     use_plus_minus = True
#                                     )
#                 dfs[f'{ds}{subgroupname}'] = res
#
#     return dfs
#

