
import os
import itertools
import multiprocessing as mp
import pandas as pd
import ast
import numpy as np
import torch
from nnunetv2.my_utils.utils import init_args, update_args_with_yaml, load_yaml_config
from nnunetv2.my_utils.utils import image_or_path_load, load_segmentation_model, chunk_ids, assign_job_numbers, \
    create_input_file, fetch_IDs, nnunet_input_output_files_list

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
        m = os.path.basename(model_dir)
        if 'MynnUNetTrainer' in m:
            name = m.split('nnUNetPlans')[0].split('MynnUNetTrainer')[1].replace('__','')
            addname = name+addname

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


















