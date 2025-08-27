
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

import seaborn as sns
import matplotlib.pyplot as plt

from nnunetv2.my_utils.utils import init_args, update_args_with_yaml, load_yaml_config, \
    load_ids_per_split, get_nnUNet_paths, get_experiments, NiftiLoader, NumpyLoader

if __name__ == "__main__":

    args = init_args()
    args = update_args_with_yaml(args, load_yaml_config(args.yml_args))

    #get nnunet locations, identify splits
    dir_pp, dir_train = get_nnUNet_paths(args.nnunet_dir, args.dataset)
    exp = get_experiments(dir_train)
    exp = exp[args.experiment]  #select experiment

    splits = load_ids_per_split(dir_pp)

    img_ldr = NumpyLoader(root_dir=os.path.join(dir_pp, 'nnUNetPlans_3d_fullres'), ID_splitter='.', incl_str='.npy')
    seg_ldr = NumpyLoader(root_dir=os.path.join(dir_pp, 'nnUNetPlans_3d_fullres'), ID_splitter='_', incl_str='_seg')

    for fold, split in splits.items():
        p_fold = os.path.join(dir_train, exp, fold)
        val_dir = os.path.join(p_fold, 'validation')
        for ID in split['val']:
            img_chans = img_ldr(ID)
            gt_chans = seg_ldr(ID)
            #select channels should be defined


            break

        #load model and do something here

        print(1)

    print(1)