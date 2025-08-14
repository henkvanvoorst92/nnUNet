
import os
import itertools
import random
import multiprocessing as mp
import pandas as pd
import ast
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import argparse
import yaml

def load_yaml_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def update_args_with_yaml(args, yaml_config):
    if args is None:
        args = argparse.Namespace()

    for key, value in yaml_config.items():
        if isinstance(value, dict):
            # Store the entire dictionary as an attribute
            setattr(args, key, value)
            # Also flatten nested dictionary keys
            for sub_key, sub_value in value.items():
                arg_key = f"{key}_{sub_key}"
                setattr(args, arg_key, sub_value)
        else:
            setattr(args, key, value)
    return args

def init_args(args=None, print_args=True):
    # Create the parser
    parser = argparse.ArgumentParser(
        description='')

    # Required positional arguments
    parser.add_argument('--yml_args', type=str, default=None,
                        help='If a path to a yml file is defined this is used to override all other args')
    parser.add_argument('--job', default=None,
                        help='list of job numbers to process')

    args = parser.parse_args(args)

    return args


def load_segmentation_model(p_seg_model, fold,
                            tile_step_size=.5,
                            checkpoint_name = 'checkpoint_final.pth',
                            gpu_id=None):
    if p_seg_model is not None:
        if os.path.exists(p_seg_model) and os.path.isdir(p_seg_model):


            predictor = nnUNetPredictor(
                tile_step_size=tile_step_size,
                use_gaussian=True,
                use_mirroring=True,
                #perform_everything_on_gpu=True,
                device=torch.device('cuda', 0) if gpu_id is None else gpu_id,
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=False
            )
            # initializes the network architecture, loads the checkpoint
            predictor.initialize_from_trained_model_folder(
                p_seg_model,
                use_folds=fold,
                checkpoint_name=checkpoint_name
            )
        else:
            print('No model found for using hence no inference possible')
        return predictor


def image_or_path_load(img_or_path):
    if isinstance(img_or_path, sitk.Image):
        out = sitk.Image(img_or_path)
    elif isinstance(img_or_path, str):
        if os.path.exists(img_or_path):
            out = sitk.ReadImage(img_or_path)
        else:
            out = None
            #raise ValueError('does not exist:',img_or_path)
    else:
        out = None
        #raise ValueError("img_or_path is not sitk.Image or a path",img_or_path)
    if out is not None:
        out = sitk.Cast(out, sitk.sitkInt16)
    return out

def chunk_ids(ids, n_jobs):
    """
    Split a list of IDs into almost equal-sized chunks.

    Parameters:
        ids (list): The list of IDs to chunk.
        n_jobs (int): The number of chunks to create.

    Returns:
        list: A list of chunks, where each chunk is a list of IDs.
    """
    chunk_size = len(ids) // n_jobs
    remainder = len(ids) % n_jobs

    random.shuffle(ids)

    chunks = []
    start = 0
    for i in range(n_jobs):
        # Distribute the remainder across the first few chunks
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(ids[start:end])
        start = end

    return chunks

def assign_job_numbers(df, num_jobs):
    num_rows = len(df)
    jobs_per_group = num_rows // num_jobs  # Number of times each job number appears
    remainder = num_rows % num_jobs  # If rows are not perfectly divisible

    job_numbers = []

    for i in range(num_jobs):
        job_numbers.extend([i] * jobs_per_group)

    # Distribute remainder evenly
    job_numbers.extend(np.random.choice(range(num_jobs), remainder, replace=False))

    np.random.shuffle(job_numbers)  # Shuffle job numbers

    df["job"] = job_numbers
    return df

def create_input_file(args, n_jobs=10, image_dirs=['NCCT-3chan', 'dwi'],
                      input_file=None, save=True, ID_splitter='_'):
    dir_inp = os.path.join(args.p_out, args.image_dir)
    IDs = list(set([f.split(ID_splitter)[0] for f in os.listdir(dir_inp) if '.nii' in f]))

    # find the local files
    dct = {}
    na_imd = []
    for imd in image_dirs:
        tmpd = {}
        p_dir = os.path.join(args.p_out, imd)
        if not os.path.exists(p_dir):
            na_imd.append(imd)
            continue
        for f in os.listdir(p_dir):
            if '.nii' in f and not ('_0001' in f or '_0002' in f):
                ID = f.split('.')[0].split(ID_splitter)[0]
                if ID in IDs:
                    file = os.path.join(p_dir, f)
                    tmpd[ID] = file if os.path.exists(file) else None
        dct[imd] = tmpd
    df = pd.DataFrame.from_dict(dct)
    for imd in na_imd:
        df[imd] = None
    df = assign_job_numbers(df, n_jobs)
    if save:
        df.to_excel(input_file)

    return df


def fetch_IDs(dir_inp):
    return list(set([f.split('_')[0] for f in os.listdir(dir_inp) if '.nii' in f]))

def segmentation_worker(inputs):

    yml_file, input_files_or_dir, output_files_or_dir, model_dir, gpu_id = inputs

    args = update_args_with_yaml(None, load_yaml_config(yml_file))

    #write something to select the IDs and pass them in dir_in

    # This function will be executed in each process
    # from imageprocess.segproc import SegmentationProcessor
    # SEGP = SegmentationProcessor(p_seg_model=model_dir,
    #                                fold=args.fold,
    #                                checkpoint_name='checkpoint_best.pth',
    #                                cli_nnunet=False,
    #                                args=args,
    #                                addname='',
    #                                overwrite=args.overwrite,
    #                                gpu_id=gpu_id)

    # SEGP.nnunet_predictor.predict_from_files(input_files_or_dir,
    #                                         output_files_or_dir,
    #                                         save_probabilities=False,
    #                                         overwrite=args.overwrite,
    #                                         num_processes_preprocessing=1,
    #                                         num_processes_segmentation_export=1,
    #                                         folder_with_segs_from_prev_stage=None,
    #                                         num_parts=1,
    #                                         part_id=0)

    if isinstance(gpu_id, int):
        gpu_id = torch.device('cuda', gpu_id)

    fold = ast.literal_eval((args.fold))

    nnunet_predictor = load_segmentation_model(model_dir, fold,
                                                tile_step_size=args.tile_step_size,
                                                checkpoint_name=args.checkpoint_name if hasattr(args, 'checkpoint_name') else 'checkpoint_best.pth',
                                                gpu_id=gpu_id)

    nnunet_predictor.predict_from_files(input_files_or_dir,
                                            output_files_or_dir,
                                            save_probabilities=False,
                                            overwrite=args.overwrite,
                                            num_processes_preprocessing=1,
                                            num_processes_segmentation_export=1,
                                            folder_with_segs_from_prev_stage=None,
                                            num_parts=1,
                                            part_id=0)



    # if args.headmask_adjust if hasattr(args, 'headmask_adjust') else False:
    #     print('Adjusting segmentation with headmask')
    #     for p_seg in output_files_or_dir:
    #         if os.path.exists(p_seg):
    #             if args.image_type== 'cta':
    #                 mask = SEGP.get_ct_headmask(crop=True)
    #             elif args.image_type== 'mra':
    #                 mask = SEGP.get_mr_brainmask(crop=False)
    #                 mask = sitk_dilate_mm(mask, kernel_mm=20)
    #             seg = SEGP.VSEGP.mask_adjust_seg(p_seg, mask)
    #             sitk.WriteImage(np2sitk(seg, mask), p_seg)


def nnunet_input_output_files_list(IDs, channels, dir_in, dir_out, overwrite=False, ID_splitter='_'):

    files_in = []
    files_out = []

    for ID in IDs:
        ID_files = []
        for chan in channels:
            for f in os.listdir(dir_in):
                #print(f, ID, chan, '.nii' in f, chan in f)
                if '.nii' in f:
                    if str(ID)==f.split(ID_splitter)[0] and chan in f:
                        ID_files.append(os.path.join(dir_in, f))
        if len(ID_files) == len(channels):
            f_out = os.path.join(dir_out, f'{ID}.nii.gz')
            if not os.path.exists(f_out) or overwrite:
                files_in.append(ID_files)
                files_out.append(f_out)
        else:
            print(f"Not all channels found for ID {ID}\n Found: {ID_files}, expected: {channels}")

    return files_in, files_out

def main_segmentation_processor(job_inputs):

    if len(job_inputs)>1:
        mp.set_start_method("spawn", force=True)
        procs = []
        for inputs in job_inputs:
            p = mp.Process(target=segmentation_worker, args=(inputs,))
            p.daemon = False   # <— make sure it can spawn its own children
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
    else:
        # If only one job, run it directly
        segmentation_worker(job_inputs[0])


if __name__ == "__main__":
    args = init_args()
    args = update_args_with_yaml(args, load_yaml_config(args.yml_args))
    addname = '_'+args.addname if hasattr(args, 'addname') else ''

    """
    args should contain:
    model_dir: root dir where models are
    models: paths to separate models
    fold: fold to use for inference (0,1,2) or all or (0,1,2,3,4)
    tile_step_size: overlap (default 0.5)
    resolution: 'full_res' etc

    dir_out: output directory --> also dir in
    gpus: list of gpu ids that are available
    n_jobs: number of jobs to run in parallel (across gpus)

    """
    if hasattr(args, 'input_file'):
        input_file = os.path.join(args.p_out, args.input_file) if os.sep not in args.input_file else args.input_file
    else:
        input_file = ''

    if hasattr(args, 'image_folders'):
        image_dirs = args.image_folders
    elif hasattr(args, 'image_dir'):
        image_dirs = [args.image_dir]
    elif hasattr(args, 'image_type'):
        image_dirs = [args.image_type]
        args.image_dir = args.image_type

    if os.path.exists(input_file):
        df = pd.read_excel(input_file, index_col=0)
    else:
        df = create_input_file(args, image_dirs=image_dirs, input_file=input_file, ID_splitter=args.ID_splitter if hasattr(args, 'ID_splitter') else '_')

    #IDs = list(set([f.split('_')[0] for f in os.listdir('/media/hvv/71672b1c-e082-495c-b560-a2dfc7d5de59/data/BL_NCCT/CRISP2/processed_june25/iat_dwi_bl_seg_june25')]))
    #df[np.isin(df.index, IDs)].to_excel(input_file)

    if args.job is not None:
        #slice the part of the IDs out that represent the job
        job = ast.literal_eval(args.job)
        df = df[np.isin(df['job'], job)]
    IDs = df.index.tolist()

    #if to many models are used reduce the size of the total jobs (otherwise processing goes x len models)
    #this distributes multiple jobs within a gpu
    if args.n_jobs < len(args.models):
        n_jobs = 1
    else:
        n_jobs = args.n_jobs // len(args.models)
    ID_chunks = chunk_ids(IDs, n_jobs)

    gpu_cycle = itertools.cycle(args.gpus)
    job_inputs = []
    pp_jobs = []
    for model, channels in args.models.items():
        model_dir = os.path.join(args.model_dir, model)

        subdir_out = '{}_{}{}'.format(args.image_dir, model.split(os.sep)[0], addname)
        for job in range(n_jobs):
            gpu_id = next(gpu_cycle)
            ID_selection = ID_chunks[job]
            dir_in = os.path.join(args.p_out, args.image_dir)
            dir_out = os.path.join(args.p_out, subdir_out)
            print(dir_in, dir_out)
            os.makedirs(dir_out, exist_ok=True)
            files_in, files_out = nnunet_input_output_files_list(ID_selection,
                                                                 channels,
                                                                 dir_in,
                                                                 dir_out,
                                                                 overwrite=args.overwrite,
                                                                 ID_splitter=args.ID_splitter if hasattr(args, 'ID_splitter') else '_'
                                                                 )
            print(files_in, files_out)

            if len(files_in) == 0:
                print(f"No files to segment for {model} in job {job}. Skipping.")
                continue
            else:
                inp = (args.yml_args, files_in, files_out, model_dir, gpu_id)
                job_inputs.append(inp)


    if len(job_inputs) > 0:
        print('Starting multiprocessing with {} jobs'.format(len(job_inputs)))
        main_segmentation_processor(job_inputs)


















