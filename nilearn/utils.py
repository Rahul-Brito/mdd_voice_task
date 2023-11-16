#Import packages
import os
import glob
import json
import pickle
import shutil
from pathlib import Path

#from tqdm import tqdm

import nilearn
import nibabel as nib
from nibabel import load
from nibabel.gifti import GiftiDataArray, GiftiImage

from nilearn import image as nimg
from nilearn import plotting as nplot
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix, run_glm
from nilearn.glm.second_level import make_second_level_design_matrix, SecondLevelModel
from nilearn.glm import fdr_threshold,threshold_stats_img
from nilearn.glm.contrasts import compute_contrast, _compute_fixed_effects_params
from bids.layout import BIDSLayout, parse_file_entities

# import cortex
# from cortex import fmriprep

from nipype.interfaces.workbench.base import WBCommand
from nipype.algorithms import modelgen
from nipype.interfaces.base import Bunch

import scipy.stats as stats

import hcp_utils as hcp

import numpy as np
import pandas as pd


def save_cifti(cifti_data, output_type, out_dir, sstr, contrast = 'spchsil', sample_cifti_file='../../derivatives/fmriprep/sub-voice857/ses-1/smoothed/sub-voice857_ses-1_task-pataka_run-1_space-fsLR_den-91k_bold.dtseries.nii'):
    #takes 91k np array of parcels and generates cifti object
    
    sample_cifti = nimg.load_img(sample_cifti_file)
    time_axis, brain_model_axis = [sample_cifti.header.get_axis(i) for i in range(sample_cifti.ndim)]
    z = np.atleast_2d(cifti_data)
    scalar_axis = nib.cifti2.ScalarAxis([output_type])  # Takes a list of names, one per row
    new_header = nib.Cifti2Header.from_axes([scalar_axis, brain_model_axis])
    #z = np.reshape(z, (-1, z.shape[0]))
    img = nib.Cifti2Image(z, new_header)

    name = f'{out_dir}/{sstr}_space-fsLR_den-91k_contrast-{contrast}_{output_type}.dscalar.nii'
    img.to_filename(name)

