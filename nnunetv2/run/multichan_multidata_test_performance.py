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
from nnunetv2.my_utils.metrics import comparative_stats, compare_multiclass_masks, compare_masks
from nnunetv2.my_utils.utils import np2sitk, image_or_path_load, sitk_dilate_mm, remove_small_cc


def create_chan_gt(ctp_gt_dir,
                   chan_gt_dir,
                   adj_seg_dir=None,
                   roi_loader=None,
                   min_cc_ml=None,
                   chans=['m6', 'm4', 'm2', 't0', 'p2', 'p4', 'p6'],
                   filext='.nii.gz',
                   dil_adj_seg=0,
):
    """
    ctp_gt_dir: input dir with 4D CTP GT files
    chan_gt_dir: output dir for channel-wise GT files
    adj_seg_dir: if path provided, files with vesselseg for adjustment are used
    roi_loader: NiftiLoader for roi files to limit evaluation region
    min_cc_ml: minimum connected component size in mL to keep in GT files
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
            gt_arr = sitk.GetArrayFromImage(gt)
            if timename in adj_file_dct:
                if ID in adj_file_dct[timename]:
                    adj_seg = sitk.ReadImage(adj_file_dct[timename][ID])
                    if dil_adj_seg>0:
                        adj_seg = sitk_dilate_mm(adj_seg, dil_adj_seg)
                    adj_seg = sitk.GetArrayFromImage(adj_seg)
                    gt_arr = gt_arr*adj_seg
                    #something with mincc and roi here?
                    if roi_loader is not None:
                        if ID in roi_loader.file_paths:
                            roi = roi_loader(ID)
                            roi_arr = sitk.GetArrayFromImage(roi)
                            gt_arr = gt_arr * roi_arr
                        else:
                            print(f'[!] No ROI for {ID}')

                    if min_cc_ml is not None:
                        voxel_volume = np.prod(gt.GetSpacing()) / 1000  # in mL
                        min_cc_voxels = int(np.ceil(min_cc_ml / voxel_volume))
                        gt_arr = gt_arr*remove_small_cc(gt_arr, min_cc_voxels)

                    gt = np2sitk(gt_arr, gt)
                else:
                    print(f'[!] No adjustment seg for {ID} time {timename}')
            sitk.WriteImage(gt, f_out)

def create_new_gt(org_gt_dir,new_gt_dir,roi_loader=None, min_cc_ml=None, addname='_cta', ID_splitter='_', filext='.nii.gz'):

    #adjust gt files by only selecting a roi and removing cc smaller than min_cc_ml
    os.makedirs(new_gt_dir, exist_ok=True)
    org_gt_ldr = NiftiLoader(root_dir=org_gt_dir, ID_splitter=ID_splitter, filext=filext)
    org_IDs = list(org_gt_ldr.file_paths.keys())
    new_IDs_available = [f.split(ID_splitter)[0] for f in os.listdir(new_gt_dir)]
    intersect_IDs =  list(set(org_IDs) - set(new_IDs_available))
    if len(intersect_IDs)==0:
        return get_path_dict(new_gt_dir, ID_splitter=ID_splitter, filext=filext)

    for ID in tqdm(org_gt_ldr.file_paths.keys()):
        f_out = os.path.join(new_gt_dir, f"{ID}{addname}.nii.gz")
        if os.path.exists(f_out):
            continue
        gt = org_gt_ldr(ID)
        gt_arr = sitk.GetArrayFromImage(gt)

        if roi_loader is not None:
            if ID in roi_loader.file_paths:
                roi = roi_loader(ID)
                roi_arr = sitk.GetArrayFromImage(roi)
                gt_arr = gt_arr * roi_arr
            else:
                print(f'[!] No ROI for {ID}')

        if min_cc_ml is not None:
            voxel_volume = np.prod(gt.GetSpacing()) / 1000  # in mL
            min_cc_voxels = int(np.ceil(min_cc_ml / voxel_volume))
            #remove small CCs
            #from nnunetv2.my_utils.segmentation_postprocessing import remove_small_connected_components
            gt_arr = gt_arr*remove_small_cc(gt_arr, min_cc_voxels)

        gt_new = np2sitk(gt_arr, gt)
        sitk.WriteImage(gt_new, f_out)

def get_chan_gt(args):

    dir_roi = getattr(args, 'roi_gt', None)
    roi_reader = NiftiLoader(dir_roi, ID_splitter='_') if dir_roi is not None else None
    roiname = '_'+os.path.basename(dir_roi) if dir_roi is not None else ''


    min_cc_ml = getattr(args, 'min_cc_ml', None)
    mincc_name = f'_minCC{min_cc_ml}mL' if dir_roi is not None else ''

    #create poorcta and normal cta labels
    org_cta_gt = getattr(args, 'cta_gt', None)
    org_poorcta_gt = getattr(args, 'poorcta_gt', None)
    new_dir_add = f'{roiname}{mincc_name}'

    if org_cta_gt is not None:
        new_cta_gt = os.path.join(os.path.dirname(org_cta_gt),'cta_gt'+new_dir_add)
        create_new_gt(org_cta_gt, new_cta_gt,
                        roi_loader=roi_reader,
                        min_cc_ml=min_cc_ml,
                        addname='_cta', ID_splitter='_',
                        filext='.nii.gz')

    if org_poorcta_gt is not None:
        new_poorcta_gt = os.path.join(os.path.dirname(org_poorcta_gt),'poorcta_gt'+new_dir_add)
        create_new_gt(org_poorcta_gt, new_poorcta_gt,
                        roi_loader=roi_reader,
                        min_cc_ml=min_cc_ml,
                        addname='_cta', ID_splitter='_',
                        filext='.nii.gz')

    #simCTA GTs
    chan_gt_dir = args.simcta_gt
    dil_adj_seg = None
    if hasattr(args, 'dil_adj_seg'):
        dil_adj_seg = args.dil_adj_seg
    if not isinstance(dil_adj_seg,list):
        dil_adj_seg = [dil_adj_seg]

    chan_dirs = []
    for das in dil_adj_seg:
        if das is not None:
            chan_gt_dir_out = f'{chan_gt_dir}_adj509_dil{das}'
        else:
            chan_gt_dir_out = chan_gt_dir
        if min_cc_ml is not None:
            chan_gt_dir_out = f'{chan_gt_dir_out}_minCC{min_cc_ml}mL'


        create_chan_gt(ctp_gt_dir=args.ctp_gt ,
                       chan_gt_dir=chan_gt_dir_out,
                       adj_seg_dir=os.path.join(os.path.dirname(args.simcta_img), 'time_averages_Dataset509_CTAvseg'),
                       chans=args.chans,
                       dil_adj_seg=das,
                       filext='.nii.gz'
                       )
        chan_dirs.append(chan_gt_dir_out)

    return new_cta_gt, new_poorcta_gt, chan_dirs

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
    main_dir_simseg = args.simcta_gt if hasattr(args, 'simcta_gt') else None
    if main_dir_simseg is not None:
        if os.path.exists(main_dir_simseg):

            dil_adj_seg = args.dil_adj_seg if hasattr(args, 'dil_adj_seg') else None
            if not isinstance(dil_adj_seg, list):
                dil_adj_seg = [dil_adj_seg]

            for das in dil_adj_seg:
                if das is None:
                    dir_simseg = main_dir_simseg
                    chan_add = '_lblCTP'
                else:
                    dir_simseg = f'{main_dir_simseg}_adj509_dil{das}'
                    chan_add = f'_dil{das}'

                tmp_dct = {}
                for chan in chans:
                    ssd = get_path_dict(dir_simseg, ID_splitter='--', filext='.nii.gz', incl_str=f'peakart-{chan}')
                    print(f'Dilation GT {das} simCTA channel {chan} number of samples:', len(ssd))
                    chan_name = chan+chan_add
                    gt_dct[chan_name] = {
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

            # simcta_dct = {}
            # for ID in IDs:
            #     tmp = {}
            #     for chan in args.chans:
            #         f_pred = os.path.join(dir_simcta, f'{ID}--peakart-{chan}.nii.gz')
            #         if os.path.exists(f_pred):
            #             tmp[chan] = f_pred
            #     if len(tmp)>0:
            #         simcta_dct[ID] = tmp


            for chan in args.chans:
                chan_files = {k.split('--')[0]:v for k,v in simcta_files.items() if f'peakart-{chan}' in k}
                #simcta_dct.setdefault(chan, {}).update(chan_files)
                pred_seg.setdefault(chan, {}).update({folder: chan_files})

        dir_poor_cta = os.path.join(args.poorcta_pred,folder)
        if os.path.exists(dir_poor_cta):
            poorcta_dct = get_path_dict(dir_poor_cta, ID_splitter='_', filext='.nii.gz')

        pred_seg.setdefault('cta', {}).update({folder: cta_dct})
        #pred_seg.setdefault('simcta', {}).update({folder: simcta_dct})
        pred_seg.setdefault('poorcta', {}).update({folder: poorcta_dct})

    return pred_seg

def get_single_segmentation_performance(gt, pred,
                                        roi=None,
                                        compute_hausdorff=False,
                                        vessel_metrics=False):
    gt = image_or_path_load(gt)
    pred = np2sitk(sitk.GetArrayFromImage(image_or_path_load(pred)), gt)
    if roi is not None:
        #roi should be headmask (or other mask) to limit evaluation intracranial vessels
        roi = image_or_path_load(roi)
        pred = np2sitk(sitk.GetArrayFromImage(pred) * sitk.GetArrayFromImage(roi), gt)
        gt = np2sitk(sitk.GetArrayFromImage(gt) * sitk.GetArrayFromImage(roi), gt)

    #separate AV segmentation
    metr_av = compare_multiclass_masks(pred, gt,
                                    compute_hausdorff=compute_hausdorff,
                                    vessel_metrics=vessel_metrics)
    out = metrics_in_df(metr_av, multiclass=True)

    #Any vessel segmented
    metr_ves = compare_masks(sitk.Cast(pred>0, sitk.sitkInt16),
                                sitk.Cast(gt>0, sitk.sitkInt16),
                            compute_hausdorff=compute_hausdorff,
                            vessel_metrics=vessel_metrics)
    out = out.merge(metrics_in_df(metr_ves), left_index=True, right_index=True, how='outer')

    return out

def prepare_jobs(gt_dct, pred_dct, n_procs=1, dir_out=None, verbose=False):


    to_process = []
    for dataset, gtd in gt_dct.items():

        pred_dataset = dataset.split('_dil')[0].split('_lblCTP')[0]

        if pred_dataset not in pred_dct:
            print(f'[!] No predictions for dataset {pred_dataset} found')
            continue

        for experiment, prd in pred_dct[pred_dataset].items():

            id_list = list(set(gtd['gt'].keys()).intersection(set(prd.keys())))

            for id in id_list:
                f_out = os.path.join(dir_out, f'{dataset}_{experiment}_{id}_results.xlsx')
                if not os.path.exists(f_out):
                    to_process.append((id,experiment,dataset, f_out))

    print(f'Total jobs to process: {len(to_process)}')
    jobs = []
    job_splits = np.array_split(to_process, n_procs)
    for ix,split in enumerate(job_splits):
        if verbose:
            split = tqdm(split, desc=f'Running job {ix+1}')
        jobs.append((split, gt_dct, pred_dct, True, True))

    return jobs


def metrics_in_df(metr, multiclass=False):

    if multiclass:
        out = pd.DataFrame(metr['per_class'])
        out= out.merge(pd.DataFrame.from_dict(metr['macro_avg'], orient='index', columns=['macro_avg']),
                        left_index=True, right_index=True, how='outer')
        out= out.merge(pd.DataFrame.from_dict(metr['micro_avg'], orient='index', columns=['micro_avg']),
                        left_index=True, right_index=True, how='outer')
        out.loc['analysis'] = out.columns
    else:
        out = pd.DataFrame.from_dict(metr, orient='index')

    return out


def seg_performance_worker(inp):

    #job should be list of (IDs, exps, datasets)
    job_batch, gt_dct, pred_dct, compute_hausdorff, vessel_metrics= inp
    for (ID,exp,dataset, f_out) in job_batch:
        try:
            pred_dataset = dataset.split('_dil')[0].split('_lblCTP')[0]
            tmp = get_single_segmentation_performance(gt_dct[dataset]['gt'][ID], pred_dct[pred_dataset][exp][ID],
                                                        roi=gt_dct[dataset]['roi'][ID],
                                                        compute_hausdorff=compute_hausdorff,
                                                        vessel_metrics=vessel_metrics).T
            tmp['res_type'] = tmp.index
            tmp['analysis'] = 'org'
            tmp['ID'] = ID
            tmp['experiment'] = exp
            tmp['dataset'] = dataset
            tmp = tmp.iloc[:, ::-1]
            tmp.to_excel(f_out, index=False)
        except:
            print(f'[!] Error processing {dataset} {exp} {ID}')

def multiprocess_seg_performance(jobs):

    torch.multiprocessing.set_start_method("spawn", force=True)
    procs = []
    for inputs in jobs:
        p = torch.multiprocessing.Process(target=seg_performance_worker, args=(inputs,))
        p.daemon = False   # <— make sure it can spawn its own children
        p.start()

if __name__ == "__main__":

    # --yml_args raw_CTP_melbourne/files/mchan_av_val.yml

    args = init_args()
    args = update_args_with_yaml(args, load_yaml_config(args.yml_args))

    #creates separate simcta eval seg files
    get_chan_gt(args)

    #make data loaders
    cta_ldr, simcta_ldr, poorcta_ldr, gt_dct = test_loaders(args)
    pred_dct = pred_seg_loaders(args)

    #prepare jobs for multiprocess performance computation
    dir_out = os.path.join(args.p_out, 'test_results_per_ID')
    os.makedirs(dir_out, exist_ok=True)
    jobs = prepare_jobs(gt_dct, pred_dct, n_procs=args.n_procs, dir_out=dir_out)
    #get perfromance results
    if len(jobs)>0:
        if args.n_procs>1:
            multiprocess_seg_performance(jobs)
        else:
            for job in jobs:
                seg_performance_worker(job)

    #fetch single results file


    #make results table
