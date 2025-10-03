
import os
import itertools
import random
import multiprocessing as mp
from multiprocessing import Pool
import pandas as pd
import ast
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from nnunetv2.my_utils.utils import init_args, update_args_with_yaml, load_yaml_config, \
    load_ids_per_split, get_nnUNet_paths, get_experiments, NiftiLoader, NumpyLoader, load_segmentation_model, \
    get_path_dict, combine_excel_files
from nnunetv2.my_utils.metrics import compare_masks, compare_multiclass_masks
from nnunetv2.my_utils.plots import all_val_plots

chan_dct = {
    't2':  [2,3,4],
    't4': [1,3,5],
    't6': [0,3,6],
    't24': [1,2,3,4,5],
    't26': [0,2,3,4,6],
    't46': [0,1,3,5,6],
    't246': [0,1,2,3,4,5,6]
}


def has_all_chans(p_dir, ID, chans):
    """
    Check if all channels are available for a given ID in the specified directory.

    Parameters:
        p_dir (str): Path to the directory containing the files.
        ID (str): The identifier for the files.
        chans (list): List of channel numbers to check.

    Returns:
        bool: True if all channels are available, False otherwise.
    """
    for chan in chans:
        file_name = f"{ID}_chan{chan}.nii.gz"
        file_path = os.path.join(p_dir, file_name)
        if not os.path.exists(file_path):
            return False
    return True


def create_val_dirs(dir_raw, splits, chans=[0,1,2,3,4,5,6]):

    dir_raw2 = dir_raw.replace('Dataset519_AV_cta', 'Dataset521_AV_timechan')

    img_ldr = NiftiLoader(root_dir=os.path.join(dir_raw2, 'imagesTr'), ID_splitter='_', incl_str='_0000.nii.gz')
    seg_ldr = NiftiLoader(root_dir=os.path.join(dir_raw2, 'labelsTr'), ID_splitter='.', incl_str='.nii.gz')

    # write all to separate folder for inference
    for fold, split in splits.items():
        raw_img_fold_dir = os.path.join(dir_raw2, 'fold_' + fold[-1], 'imagesVal')
        raw_seg_fold_dir = os.path.join(dir_raw2, 'fold_' + fold[-1], 'labelsVal')
        os.makedirs(raw_img_fold_dir, exist_ok=True)
        os.makedirs(raw_seg_fold_dir, exist_ok=True)
        for ID in tqdm(split['val']):
            has_imgs = has_all_chans(raw_img_fold_dir, ID, chans)
            has_segs = has_all_chans(raw_seg_fold_dir, ID, chans)
            if not (has_imgs and has_segs):
                imgs = img_ldr(ID)
                segs = seg_ldr(ID)
                for i in range(imgs.GetSize()[-1]):
                    sitk.WriteImage(imgs[...,i], os.path.join(raw_img_fold_dir, f"{ID}_chan{i}.nii.gz"))
                    sitk.WriteImage(segs[...,i], os.path.join(raw_seg_fold_dir, f"{ID}_chan{i}.nii.gz"))

def nested_val_img_gt_loaders(dir_raw, folds=[0,1,2,3,4], chans=[0,1,2,3,4,5,6]):

    #dir_raw = dir_raw.replace('Dataset519_AV_cta', 'Dataset521_AV_timechan')

    dct1 = {}
    for fold in folds:
        dct2 = {}
        for chan in chans:
            img_ldr = get_path_dict(os.path.join(dir_raw, f'fold_{fold}', 'imagesVal'), ID_splitter='_', filext='.nii.gz', incl_str=f'_chan{chan}')
            seg_ldr = get_path_dict(os.path.join(dir_raw, f'fold_{fold}', 'labelsVal'), ID_splitter='_', filext='.nii.gz', incl_str=f'_chan{chan}')
            dct2[chan] = {'img': img_ldr, 'gt': seg_ldr}
        dct1[f'fold_{fold}'] = dct2
    return dct1

def validation_batches(experiments, overwrite=False, only_fold=None, results_mode=False):

    num_gpus = torch.cuda.device_count()
    gpu_cycle = itertools.cycle([f"cuda:{i}" for i in range(num_gpus)])

    val_jobs = []
    files_out = []
    data = []
    for name_exp, exp in experiments.items():
        #exp = os.path.basename(exp)
        for fold, split in splits.items():
            if only_fold is not None:
                if fold!=only_fold:
                    continue
            gpu_id = next(gpu_cycle)
            p_exp = exp #os.path.join(dir_train, exp)
            p_fold = os.path.join(p_exp, fold)
            foldno = int(fold[-1])
            dir_val = os.path.join(p_fold, 'validation')
            os.makedirs(dir_val, exist_ok=True)
            files_in, files_out = [], []
            for ID in split['val']:
                imgs_in = [[dct[fold][c]['img'][ID]] for c in [0, 1, 2, 3, 4, 5, 6]]
                segs_out = [os.path.join(dir_val,os.path.basename(f[0])) for f in imgs_in]
                out_exists = all([os.path.exists(f) for f in segs_out])
                if (out_exists and (not overwrite) and (not results_mode)):
                    continue
                files_in.extend(imgs_in)
                files_out.extend(segs_out)
            val_jobs.append((
                files_in, files_out, p_exp, foldno, gpu_id, overwrite
            ))
            tmp = pd.DataFrame(files_out, columns=['file'])
            tmp['fold'] = f'fold_{foldno}'
            tmp['experiment'] = name_exp
            tmp['ID'] = [os.path.basename(f).split('.')[0].split('_')[0] for f in files_out]
            tmp['channel'] = [int(os.path.basename(f).split('_chan')[-1].split('.')[0]) for f in files_out]
            data.append(tmp)

    data = pd.concat(data)
    nested_dict = {}
    for (experiment, fold, channel, ID), group in data.groupby(['experiment', 'fold', 'channel', 'ID']):
        nested_dict.setdefault(experiment, {}).setdefault(fold, {}).setdefault(channel, {})[ID] = group['file'].values[0]

    return val_jobs, nested_dict

def results_worker(job):
    dir_out, gt_dct, seg_dct, exp, fold, chan, compute_hausdorff = job

    p_pc = os.path.join(dir_out, f'results_per_class_{exp}_{fold}_chan{chan}.xlsx')
    p_mm = os.path.join(dir_out, f'results_macro_micro_{exp}_{fold}_chan{chan}.xlsx')

    if not (os.path.exists(p_pc) and os.path.exists(p_mm)):
        # if chan in seg_dct[exp][fold].keys():
        pc, mm = [], []
        for ID, p_seg in seg_dct[exp][fold][chan].items():
            if not ID in gt_dct[fold][chan]['gt'].keys():
                continue

            p_seg = seg_dct[exp][fold][chan][ID]
            p_gt = gt_dct[fold][chan.split('_')[0]]['gt'][ID]
            p_roi = gt_dct[fold][chan.split('_')[0]]['roi'][ID]

            res = compare_multiclass_masks(p_seg, p_gt, p_roi, compute_hausdorff=compute_hausdorff)

            # Convert per_class to DataFrame
            per_class_df = pd.DataFrame(res['per_class']).T
            per_class_df.index.name = 'Class'
            per_class_df['ID'] = ID
            per_class_df['experiment'] = exp
            per_class_df['fold'] = fold
            per_class_df['channel'] = chan

            # Convert macro_avg and micro_avg to DataFrame
            macro_micro_df = pd.DataFrame([res['macro_avg'], res['micro_avg']],
                                          index=['macro_avg', 'micro_avg']).reset_index()
            macro_micro_df['ID'] = ID
            macro_micro_df['experiment'] = exp
            macro_micro_df['fold'] = fold
            macro_micro_df['channel'] = chan

            pc.append(per_class_df)
            mm.append(macro_micro_df)

        if len(pc)>0:
            pc = pd.concat(pc).reset_index()
            pc.to_excel(p_pc, index=False)

        if len(mm)>0:
            mm = pd.concat(mm).reset_index()
            mm.to_excel(p_mm, index=False)

def get_all_results(gt_dct, seg_dct, compute_hausdorff=True):

    pc, mm = [], []
    for exp,d1 in seg_dct.items():
        for fold, d2 in d1.items():
            for chan, d3 in d2.items():
                for ID, p_seg in tqdm(d3.items()):
                    p_seg = seg_dct[exp][fold][chan][ID]
                    p_gt = gt_dct[fold][chan]['gt'][ID]
                    res = compare_multiclass_masks(p_seg, p_gt, compute_hausdorff=compute_hausdorff)

                    # Convert per_class to DataFrame
                    per_class_df = pd.DataFrame(res['per_class']).T
                    per_class_df.index.name = 'Class'
                    per_class_df['ID'] = ID
                    per_class_df['experiment'] = exp
                    per_class_df['fold'] = fold
                    per_class_df['channel'] = chan

                    # Convert macro_avg and micro_avg to DataFrame
                    macro_micro_df = pd.DataFrame([res['macro_avg'], res['micro_avg']],
                                                  index=['macro_avg', 'micro_avg']).reset_index()
                    macro_micro_df['ID'] = ID
                    macro_micro_df['experiment'] = exp
                    macro_micro_df['fold'] = fold
                    macro_micro_df['channel'] = chan

                    pc.append(per_class_df)
                    mm.append(macro_micro_df)

    pc = pd.concat(pc).reset_index()
    mm = pd.concat(mm).reset_index()

    return pc, mm

def create_results_jobs(dir_out, gt_dct, seg_dct, compute_hausdorff=True, overwrite=False):
    jobs = []
    for exp,d1 in seg_dct.items():
        for fold, d2 in d1.items():
            for chan, d3 in d2.items():

                p_pc = os.path.join(dir_out, f'results_per_class_{exp}_{fold}_chan{chan}.xlsx')
                p_mm = os.path.join(dir_out, f'results_macro_micro_{exp}_{fold}_chan{chan}.xlsx')
                if os.path.exists(p_pc) and os.path.exists(p_mm):
                    continue
                jobs.append((dir_out, gt_dct, seg_dct, exp, fold, chan, compute_hausdorff))

    return jobs

def main_results_processor(dir_out, gt_dct, seg_dct, compute_hausdorff=True, n_procs=1, overwrite=False):

    os.makedirs(dir_out, exist_ok=True)

    job_inputs = create_results_jobs(dir_out, gt_dct, seg_dct, compute_hausdorff, overwrite=overwrite)
    if len(job_inputs)>0:
        if n_procs>1:
            with Pool(processes=n_procs) as pool:
                pool.map(results_worker, job_inputs)
        else:
            for inputs in job_inputs:
                results_worker(inputs)

def worker(job):
    files_in, files_out, exp, fold, gpu_id, overwrite = job

    print('Running:', os.path.basename(exp), fold, gpu_id, len(files_in), len(files_out))
    print('Input files:', files_in[0])
    print('Output files:', files_out[0])

    nnunet_predictor = load_segmentation_model(
                                p_seg_model=exp,
                                fold=[fold] if not (isinstance(fold, list) or isinstance(fold,tuple)) else fold,
                                tile_step_size=args.tile_step_size if hasattr(args, 'tile_step_size') else 0.75,
                                checkpoint_name='checkpoint_best.pth',
                                gpu_id=torch.device(gpu_id)
                            )

    nnunet_predictor.predict_from_files(files_in,
                                files_out,
                                save_probabilities=False,
                                overwrite=overwrite,
                                num_processes_preprocessing=8,
                                num_processes_segmentation_export=8,
                                folder_with_segs_from_prev_stage=None,
                                num_parts=1,
                                part_id=0)


def main_processor(job_inputs, n_procs=False):

    if n_procs>1:
        with Pool(processes=4) as pool:
            pool.map(worker, job_inputs)
    else:
        for inputs in job_inputs:
            worker(inputs)

def mchan_val_results(p_out):

    pc_all_file = os.path.join(p_out, 'mchan_nnunet_val_results', 'results_per_class.pic')
    mm_all_file = os.path.join(p_out, 'mchan_nnunet_val_results', 'results_macro_micro.pic')

    if not (os.path.exists(pc_all_file) and os.path.exists(mm_all_file)):
        pc = combine_excel_files(os.path.join(p_out, 'mchan_nnunet_val_results'), 'results_per_class_')
        mm = combine_excel_files(os.path.join(p_out, 'mchan_nnunet_val_results'), 'results_macro_micro_')
        pc.to_pickle(pc_all_file)
        mm.to_pickle(mm_all_file)
    else:
        pc = pd.read_pickle(pc_all_file)
        mm = pd.read_pickle(mm_all_file)

    return pc, mm





if __name__ == "__main__":

    args = init_args()
    args = update_args_with_yaml(args, load_yaml_config(args.yml_args))

    #get nnunet locations, identify splits
    dir_raw, dir_pp, dir_train = get_nnUNet_paths(args.nnunet_dir, args.dataset)
    experiments = get_experiments(dir_train)
    experiments = {k: v for k, v in experiments.items() if
                   k in args.experiment or k == args.experiment or any([exp in k for exp in args.experiment])}

    splits = load_ids_per_split(dir_pp)

    create_val_dirs(dir_raw, splits, chans=[0, 1, 2, 3, 4, 5, 6])

    dct = nested_val_img_gt_loaders(dir_raw, folds=[0, 1, 2, 3, 4], chans=[0, 1, 2, 3, 4, 5, 6])

    only_fold = 'fold_{}'.format(args.fold) if args.fold is not None else None
    res_mode = args.compute_results_mode if hasattr(args, 'compute_results_mode') else False
    val_jobs, seg_dct = validation_batches(experiments, overwrite=args.overwrite, only_fold=only_fold, results_mode=res_mode)

    if res_mode:
        main_results_processor(os.path.join(args.p_out, 'mchan_nnunet_val_results'),
                               dct, seg_dct,
                               compute_hausdorff=True,
                               n_procs=args.n_procs,
                               overwrite=args.overwrite)
        pc, mm = mchan_val_results(args.p_out)
        all_val_plots(pc, dir_figs=os.path.join(args.p_out, 'mchan_nnunet_figures'), addname='val_results_', select_exp=['t0', 't2', 't6'])
        print(1)
    else:
        main_processor(val_jobs, n_procs=args.n_procs)


    print(1)
    # print('Validation for:', exp, fold)
    # print('Input files:', files_in)
    # print('Output files:', files_out)
    #
    # nnunet_predictor.predict_from_files(files_in,
    #                                     files_out,
    #                                     save_probabilities=False,
    #                                     overwrite=args.overwrite,
    #                                     num_processes_preprocessing=1,
    #                                     num_processes_segmentation_export=1,
    #                                     folder_with_segs_from_prev_stage=None,
    #                                     num_parts=1,
    #                                     part_id=0)
