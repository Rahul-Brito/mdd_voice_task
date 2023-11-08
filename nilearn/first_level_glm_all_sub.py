#Import packages
import os
import glob
import json
#from tqdm import tqdm

import nilearn
import nibabel as nib
from nilearn import image as nimg
from nilearn import plotting as nplot
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix, run_glm
from nilearn.glm import fdr_threshold,threshold_stats_img
from nilearn.glm.contrasts import compute_contrast


from bids.layout import BIDSLayout, parse_file_entities

# import cortex
# from cortex import fmriprep

from nipype.interfaces.workbench.base import WBCommand
from nipype.algorithms import modelgen
from nipype.interfaces.base import Bunch

import hcp_utils as hcp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
#! module load openmind/hcp-workbench/1.2.3

import first_level_sparse_scanning as flss 
out_dir='../../derivatives/first_level_102523'
events_path_pattern = f'/nese/mit/group/sig/om_projects/voice/bids/data/sub-voice*/ses-*/func/*events.tsv'
pop_events = flss.find_populated_events(events_path_pattern)
count_pop_events = pop_events.groupby('task').populated.value_counts()

valid_runs = pop_events.events_file[pop_events.populated == True]
parsed_valid_runs = [parse_file_entities(vr) for vr in valid_runs]

subjects_excluded = ['voice997', 'voice897','voice863'] #for some reason these one is not in the fmriprep output
parsed_valid_runs = [pvr for pvr in parsed_valid_runs if pvr['subject'] not in subjects_excluded]

low_acompcor_to_drop = flss.find_low_acompcor(parsed_valid_runs)
parsed_valid_runs = [r for r in parsed_valid_runs if r not in low_acompcor_to_drop]

tasks_included = ['pataka', 'emosent', 'vowel', 'nwr']
parsed_valid_runs = [r for r in parsed_valid_runs if any(t == r['task'] for t in tasks_included)]

flss.convolve_sparse_scan_glm_with_cifti(parsed_valid_runs, out_dir)
#first_level_stats_maps_df = 
#first_level_stats_maps_df.to_pickle('102523_effects_size.pkl')