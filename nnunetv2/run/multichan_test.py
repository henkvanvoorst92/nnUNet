import os
import itertools
import pandas as pd
import ast
import numpy as np
import torch
import matplotlib.pyplot as plt

from nnunetv2.my_utils.utils import init_args, update_args_with_yaml, load_yaml_config, \
    get_nnUNet_paths, get_experiments, NiftiLoader, get_path_dict, combine_excel_files
from nnunetv2.run.multichan_val import main_processor, main_results_processor
from nnunetv2.my_utils.plots import boxplot_per_class

def test_job(experiments,
             img_loaders,
             fold,
             overwrite=False,
             results_mode=False):

    subset_len = 200
    data = []
    if args.input_file is not None:
        file = os.path.join(args.p_out, args.input_file) if not os.path.exists(os.path.dirname(args.input_file)) else args.input_file
        if os.path.exists(args.input_file):
            data = pd.read_excel(args.input_file)
    if len(data)==0:
        num_gpus = torch.cuda.device_count()
        gpu_cycle = itertools.cycle([f"cuda:{i}" for i in range(num_gpus)])

        #dir_test = os.path.join(args.p_out, 'test_segs')
        #os.makedirs(dir_test, exist_ok=True)

        for name_exp, p_exp in experiments.items():
            gpu_id = next(gpu_cycle)
            #p_exp = os.path.join(dir_train, exp)


            imgs_in, segs_out = [], []
            for ldr in img_loaders:
                imgs = list(ldr.file_paths.values())
                imgs_in.extend(imgs)
                pred_ldr = pred_loader(p_exp, os.path.dirname(imgs[0]))
                try:
                    segs_out.extend([pred_ldr[ID] for ID in ldr.file_paths.keys()])
                except:
                    print(1)

            out_exists = all([os.path.exists(f) for f in segs_out])
            if (out_exists and (not overwrite) and (not results_mode)):
                continue

            tmp = pd.DataFrame(segs_out, columns=['files_out'])
            tmp['files_in'] = imgs_in
            tmp['fold'] = fold
            tmp['gpu_id'] = gpu_id
            tmp['overwrite'] = overwrite
            tmp['experiment'] = name_exp
            tmp['p_exp'] = p_exp
            tmp['ID'] = [os.path.basename(f).split('.')[0].split('_')[0].split('--')[0] for f in segs_out]
            tmp['channel'] = [os.path.basename(f).split('peakart-')[-1].split('.')[0] if 'peakart-' in os.path.basename(f) else 'cta' for f in segs_out]
            data.append(tmp)

        data = pd.concat(data)
        data['job'] = np.random.randint(0, 10, size=len(data))
        data.to_excel(file, index=False)

    if args.job is not None:
        job = ast.literal_eval(args.job)
        data = data[np.isin(data['job'], job)]

    test_jobs = []
    for __,row in data.iterrows():
        test_jobs.append((row['files_in'], row['files_out'], row['p_exp'], ast.literal_eval(row['fold']), row['gpu_id'], row['overwrite']))

    nested_dict = {}
    for (experiment, fold, channel, ID), group in data.groupby(['experiment', 'fold', 'channel', 'ID']):
        nested_dict.setdefault(experiment, {}).setdefault(fold, {}).setdefault(channel, {})[ID] = group['files_out'].values[0]

    return test_jobs, nested_dict

def test_loaders(args):

    img_ldr, seg_ldr = None, None
    sim_ldr, simseg_ldr = None, None
    roi_ldr = None

    #real cta image loader (for inference)
    dir_cta = args.cta_img if hasattr(args, 'cta_img') else None
    if dir_cta is not None:
        if os.path.exists(dir_cta):
            img_ldr = NiftiLoader(dir_cta, ID_splitter='_')
    #ctps to generate ctas image loader (for inference)
    dir_sim = args.simcta_img if hasattr(args, 'simcta_img') else None
    if dir_sim is not None:
        if os.path.exists(dir_sim):
            sim_ldr = NiftiLoader(dir_sim, ID_splitter='_', incl_str='peakart')
            simdata = pd.DataFrame(data=[sim_ldr.file_paths.values(), sim_ldr.file_paths.keys()], index=['file_path', 'mID']).T
            simdata['chan'] = [mID.split('peakart-')[-1].split('.')[0] for mID in simdata['mID']]
            simdata.index = [mID.split('--')[0] for mID in simdata['mID']]
            chans = list(set(simdata['chan']))

            img_chan = {}
            for chan in chans:
                img_chan[chan] = simdata[simdata['chan']==chan]['file_path'].to_dict()

    dir_roi = args.roi_gt if hasattr(args, 'roi_gt') else None
    if dir_roi is not None:
        if os.path.exists(dir_roi):
            roi_ldr = NiftiLoader(dir_roi, ID_splitter='_')

    #gt structure to compute results
    gt_dct = {} #all paths to GT data
    dir_gt = args.cta_gt if hasattr(args, 'cta_gt') else None
    if dir_gt is not None:
        if os.path.exists(dir_gt):
            seg_ldr = get_path_dict(dir_gt, ID_splitter='_', filext='.nii.gz')
            imlr = {k:v for k,v in img_ldr.file_paths.items() if k in seg_ldr.keys()}
            imdct = {'cta': {'gt': seg_ldr,
                             'img': imlr,
                             'roi': roi_ldr.file_paths if roi_ldr is not None else None}
                     }
            gt_dct.update({args.fold:imdct})

    dir_simseg = args.simcta_gt if hasattr(args, 'simcta_gt') else None
    if dir_simseg is not None:
        if os.path.exists(dir_simseg):
            #simseg_ldr = NiftiLoader(dir_simseg, ID_splitter='_', incl_str='peakart')
            simdct = {}
            for chan in chans:
                ssd = get_path_dict(dir_simseg, ID_splitter='--', filext='.nii.gz', incl_str=f'peakart-{chan}')
                simdct[chan] = {
                                'gt': ssd,
                                'img': img_chan[chan],
                                'roi':roi_ldr.file_paths if roi_ldr is not None else None
                                }
                gt_dct.update({args.fold: {**simdct, **gt_dct[args.fold]}})

    return img_ldr, sim_ldr, gt_dct

def get_test_experiments(args):
    # get nnunet locations, identify splits
    dataset = [args.dataset] if isinstance(args.dataset,str) else args.dataset

    exps = {}
    for ds in dataset:
        dir_raw, dir_pp, dir_train = get_nnUNet_paths(args.nnunet_dir, ds)
        experiments = get_experiments(dir_train)
        exps.update(experiments)

    #select experiments
    if args.experiment is not None:
        experiments = {}
        for k,v in exps.items():
            expname = k.split('_')[0]
            if expname in args.experiment or expname==args.experiment:
                experiments[k] = v
    else:
        experiments = exps

    return experiments

def pred_loader(p_exp, img_dir):

    [exp,m] = p_exp.split(os.sep)[-2:]
    if 'MynnUNetTrainer' in m:
        name = m.split('nnUNetPlans')[0].split('MynnUNetTrainer')[1].replace('__', '')
    else:
        if 'lblCTP' in exp:
            name = f'_lblCTP'
        else:
            name = ''

    seg_dir = os.path.join(os.path.dirname(img_dir), '{}_{}{}'.format(os.path.basename(img_dir), exp, name))
    if os.path.exists(seg_dir):
        out = get_path_dict(seg_dir, ID_splitter='_', filext='.nii.gz')
    else:
        os.makedirs(seg_dir, exist_ok=True)
        out = {}
        for f in os.listdir(img_dir):
            ID = f.split('.')[0].split('_')[0]
            out[ID] = os.path.join(seg_dir, ID+'.nii.gz')

    return out

def mchan_test_results(p_out):

    pc_all_file = os.path.join(p_out, 'test_results','results_per_class.pic')
    mm_all_file = os.path.join(p_out, 'test_results','results_macro_micro.pic')

    if not (os.path.exists(pc_all_file) and os.path.exists(mm_all_file)):
        pc = combine_excel_files(os.path.join(p_out, 'test_results'), 'results_per_class_')
        mm = combine_excel_files(os.path.join(p_out, 'test_results'), 'results_macro_micro_')
        pc.to_pickle(pc_all_file)
        mm.to_pickle(mm_all_file)
    else:
        pc = pd.read_pickle(pc_all_file)
        mm = pd.read_pickle(mm_all_file)

    return pc, mm

def test_figures(pc, args):

    dir_fig_cta = os.path.join(args.p_out, 'figures', 'cta')
    os.makedirs(dir_fig_cta, exist_ok=True)
    dir_fig_4d = os.path.join(args.p_out, 'figures', '4d')
    os.makedirs(dir_fig_4d, exist_ok=True)

    #rename classes for plot
    pc['Class'] = pc['Class'].map({1: 'Artery', 2: 'Vein', 3: 'Both'})
    #rename experiments
    #df['fruit'] = pd.Categorical(df['fruit'], categories=desired_order, ordered=True)
    # Now sort by that column
    #df_sorted = df.sort_values('fruit')


    outcomes = ['Dice', 'TPR', 'FPR', 'PPV', 'NPV',
                'pred-gt_VD', 'AVD', 'Hausdorff', 'HD95', 'AHD']
    for outcome in outcomes:
        boxplot_per_class(pc[pc['channel'] == 'cta'],
                          y=outcome, x='experiment',
                          subplot_by='Class',
                          save_path=os.path.join(dir_fig_cta, f'boxplot_cta_{outcome}.png')
                          )

    return


if __name__ == "__main__":

    # --yml_args raw_CTP_melbourne/files/mchan_av_val.yml

    args = init_args()
    args = update_args_with_yaml(args, load_yaml_config(args.yml_args))


    experiments = get_test_experiments(args)
    # select experiment
    res_mode = args.compute_results_mode if hasattr(args, 'compute_results_mode') else False

    cta_ldr, simcta_ldr, gt_dct = test_loaders(args)

    jobs, seg_dct = test_job(experiments,
                                img_loaders=(cta_ldr, simcta_ldr),
                                fold=args.fold,
                                overwrite=args.overwrite,
                                results_mode=res_mode)

    if res_mode:
        main_results_processor(os.path.join(args.p_out, 'test_results'),
                               gt_dct, seg_dct,
                               compute_hausdorff=True,
                               n_procs=args.n_procs,
                               overwrite=args.overwrite)
        pc, mm = mchan_test_results(args.p_out)

        test_figures(pc, args)
    else:
        main_processor(jobs, n_procs=args.n_procs)
