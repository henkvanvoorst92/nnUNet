import os
import itertools
import pandas as pd
import ast
import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm
import matplotlib.pyplot as plt

from nnunetv2.my_utils.utils import init_args, update_args_with_yaml, load_yaml_config, \
    get_nnUNet_paths, get_experiments, NiftiLoader, get_path_dict, combine_excel_files, np2sitk, write_multitab_excel
from nnunetv2.run.multichan_val import main_processor, main_results_processor
from nnunetv2.my_utils.plots import boxplot_per_class, test_time_plots
from nnunetv2.my_utils.metrics import comparative_stats

def test_job(experiments,
             img_loaders,
             fold,
             overwrite=False,
             results_mode=False):
    os.makedirs(args.p_out, exist_ok=True)

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
                pred_ldr = pred_loader(p_exp, os.path.dirname(imgs[0]))
                try:
                    segs_out.extend([pred_ldr[ID] for ID in ldr.file_paths.keys()])
                except:
                    print(name_exp, 'prediction loader error')
                    continue
                imgs_in.extend(imgs)

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
    poorcta_ldr, poorseg_ldr = None, None
    roi_ldr = None
    gt_dct = {}

    #real cta image loader (for inference)
    dir_cta = args.cta_img if hasattr(args, 'cta_img') else None
    if dir_cta is not None:
        if os.path.exists(dir_cta):
            img_ldr = NiftiLoader(dir_cta, ID_splitter='_')

    #ctps to generate ctas image loader (for inference) --> should already be split per frame with filename including --peakart-m4_
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

    dir_poor = args.poorcta_img if hasattr(args, 'poorcta_img') else None
    if dir_poor is not None:
        if os.path.exists(dir_cta):
            poorcta_ldr = NiftiLoader(dir_cta, ID_splitter='_')

    #add region of interest used for evaluation
    dir_roi = args.roi_gt if hasattr(args, 'roi_gt') else None
    if dir_roi is not None:
        if os.path.exists(dir_roi):
            roi_ldr = NiftiLoader(dir_roi, ID_splitter='_')

    #original cta dataset
    dir_seg = args.cta_gt if hasattr(args, 'cta_gt') else None
    if dir_seg is not None:
        if os.path.exists(dir_seg):
            seg_ldr = get_path_dict(dir_seg, ID_splitter='_', filext='.nii.gz')
            imlr = {k:v for k,v in img_ldr.file_paths.items() if k in seg_ldr.keys()}
            print('Original CTA dataset number of samples:',len(seg_ldr))
            gt_dct['cta'] =  {'gt': seg_ldr,
                             'img': imlr,
                             'roi': roi_ldr.file_paths if roi_ldr is not None else None}

    #ground truth segmentations for simulated cta --> should be per frame
    dir_simseg = args.simcta_gt if hasattr(args, 'simcta_gt') else None
    if dir_simseg is not None:
        if os.path.exists(dir_simseg):
            #simseg_ldr = NiftiLoader(dir_simseg, ID_splitter='_', incl_str='peakart')
            for chan in chans:
                ssd = get_path_dict(dir_simseg, ID_splitter='--', filext='.nii.gz', incl_str=f'peakart-{chan}')
                print(f'simCTA channel {chan} number of samples:', len(ssd))
                gt_dct[chan] = {
                                'gt': ssd,
                                'img': img_chan[chan],
                                'roi':roi_ldr.file_paths if roi_ldr is not None else None
                                }

    dir_poorseg = args.poorcta_gt if hasattr(args, 'poorcta_gt') else None
    if dir_poorseg is not None:
        if os.path.exists(dir_poorseg):
            poorseg_ldr = get_path_dict(dir_poorseg, ID_splitter='_', filext='.nii.gz')
            imlr = {k:v for k,v in poorcta_ldr.file_paths.items() if k in poorseg_ldr.keys()}
            print('Poor CTA dataset number of samples:', len(poorseg_ldr))
            gt_dct['poorcta'] =  {'gt': poorseg_ldr,
                             'img': imlr,
                             'roi': roi_ldr.file_paths if roi_ldr is not None else None}

    return img_ldr, sim_ldr, poorcta_ldr, gt_dct


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




def create_chan_gt(ctp_gt_dir,
                   chan_gt_dir,
                   adj_seg_dir=None,
                   chans=['m6', 'm4', 'm2', 't0', 'p2', 'p4', 'p6'],
                   filext='.nii.gz'
):
    """
    ctp_gt_dir: input dir with 4D CTP GT files
    chan_gt_dir: output dir for channel-wise GT files
    adj_seg_dir: if path provided, files with vesselseg for adjustment are used
    chans: dict with key=str in pred and adj_seg dir files, value=channel index in 4D GT file

    """

    os.makedirs(chan_gt_dir, exist_ok=True)

    gt_ctp_ldr = NiftiLoader(root_dir=ctp_gt_dir, ID_splitter='.', filext=filext)

    adj_file_dct = {}
    if adj_seg_dir is not None:
        for timename in chans:
            adj_file_dct[timename] = get_path_dict(adj_seg_dir, ID_splitter='--', filext='.nii.gz', incl_str=timename)

    for ID in tqdm(gt_ctp_ldr.file_paths.keys()):
        for timename in chans:
            f_out = os.path.join(chan_gt_dir, f"{ID}--peakart-{timename}.nii.gz")
            if os.path.exists(f_out):
                continue
            gt = gt_ctp_ldr(ID)
            if timename in adj_file_dct:
                if ID in adj_file_dct[timename]:
                    adj_seg = sitk.GetArrayFromImage(sitk.ReadImage(adj_file_dct[timename][ID]))
                    gt = np2sitk(sitk.GetArrayFromImage(gt)*adj_seg, gt)
                else:
                    print(f'[!] No adjustment seg for {ID} time {timename}')
            sitk.WriteImage(gt, f_out)

def pred_seg_loaders(args):

    pred_seg = {}
    for folder,name in args.experiments.items():
        cta_dct, simcta_dct, poorcta_dct = {}, {}, {}

        dir_cta = os.path.join(args.cta_pred, folder)
        if os.path.exists(dir_cta):
            cta_dct = get_path_dict(os.path.join(args.cta_pred,folder), ID_splitter='_', filext='.nii.gz')

        dir_simcta = os.path.join(args.simcta_pred, f'time_averages{folder[3:]}')
        if os.path.exists(dir_simcta):
            simcta_files = get_path_dict(dir_simcta, ID_splitter='_', filext='.nii.gz')
            IDs = [k.split('--')[0] for k in simcta_files.keys()]

            simcta_dct = {}
            for ID in IDs:
                tmp = {}
                for chan in args.chans:
                    f_pred = os.path.join(dir_simcta, f'{ID}--peakart-{chan}.nii.gz')
                    if os.path.exists(f_pred):
                        tmp[chan] = f_pred
                if len(tmp)>0:
                    simcta_dct[ID] = tmp

        dir_poor_cta = os.path.join(args.poorcta_pred,folder)
        if os.path.exists(dir_poor_cta):
            poorcta_dct = get_path_dict(dir_poor_cta, ID_splitter='_', filext='.nii.gz')

        pred_seg[folder] = {
            'cta': cta_dct,
            'simcta': simcta_dct,
            'poorcta': poorcta_dct
        }

    return pred_seg

if __name__ == "__main__":

    # --yml_args raw_CTP_melbourne/files/mchan_av_val.yml

    args = init_args()
    args = update_args_with_yaml(args, load_yaml_config(args.yml_args))

    # select experiment
    res_mode = args.compute_results_mode if hasattr(args, 'compute_results_mode') else False
    if args.lbl_adjust:
        create_chan_gt(ctp_gt_dir=os.path.join(os.path.dirname(args.p_out), 'annotate/SU_CTP_todo/final_seg_daniel') ,
                       chan_gt_dir=os.path.join(os.path.dirname(args.p_out), 'annotate/SU_CTP_todo/pertime_gt_adj509'),
                       adj_seg_dir=os.path.join(os.path.dirname(args.simcta_img), 'time_averages_Dataset509_CTAvseg'),
                       chans=args.chans,
                       filext='.nii.gz'
                       )
    else:
        create_chan_gt(ctp_gt_dir=os.path.join(os.path.dirname(args.p_out), 'annotate/SU_CTP_todo/final_seg_daniel') ,
                       chan_gt_dir=os.path.join(os.path.dirname(args.p_out), 'annotate/SU_CTP_todo/pertime_gt_noadj'),
                       adj_seg_dir=None,
                       chans=args.chans,
                       filext='.nii.gz'
                       )

    cta_ldr, simcta_ldr, poorcta_ldr, gt_dct = test_loaders(args)
    pred_dct = pred_seg_loaders(args)

    #pass 2 dicts with exp and gt for eval computation


    jobs, seg_dct = test_job(experiments,
                                img_loaders=(cta_ldr, simcta_ldr),
                                fold=args.fold,
                                overwrite=args.overwrite,
                                results_mode=res_mode)

    if res_mode:
        pc_all_file = os.path.join(args.p_out, 'test_results', 'results_per_class.pic')
        mm_all_file = os.path.join(args.p_out, 'test_results', 'results_macro_micro.pic')
        if os.path.exists(pc_all_file) and os.path.exists(mm_all_file):
            pc = pd.read_pickle(pc_all_file)
            mm = pd.read_pickle(mm_all_file)
        else:
            main_results_processor(os.path.join(args.p_out, 'test_results'),
                                   gt_dct, seg_dct,
                                   compute_hausdorff=True,
                                   n_procs=args.n_procs,
                                   overwrite=args.overwrite)
            pc, mm = mchan_test_results(args.p_out)

        #optional select subset of pc of interest
        pc = pc[np.isin(pc['experiment'], ['t0','t246']) & np.isin(pc['channel'], ['cta','m6','m4','m2','t0','p2','p4','p6'])]
        stat_res = comparative_stats(pc)
        write_multitab_excel(stat_res, os.path.join(args.p_out,  'comparative_stats.xlsx'), index=True)

        test_figures(pc, args)
    else:
        main_processor(jobs, n_procs=args.n_procs)
