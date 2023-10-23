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

import first_level_sparse_scanning


events_path_pattern = f'/nese/mit/group/sig/om_projects/voice/bids/data/sub-voice*/ses-*/func/*events.tsv'
pop_events = find_populated_events(events_path_pattern)
count_pop_events = pop_events.groupby('task').populated.value_counts()

valid_runs = pop_events.events_file[pop_events.populated == True]
parsed_valid_runs = [parse_file_entities(vr) for vr in valid_runs]

subjects_excluded = ['voice997', 'voice897','voice863'] #for some reason these one is not in the fmriprep output
parsed_valid_runs = [pvr for pvr in parsed_valid_runs if pvr['subject'] not in subjects_excluded]

low_acompcor_to_drop = find_low_acompcor(parsed_valid_runs)
parsed_valid_runs = [r for r in parsed_valid_runs if r not in low_acompcor_to_drop]

tasks_included = ['pataka', 'emosent', 'vowel', 'nwr']
parsed_valid_runs = [r for r in parsed_valid_runs if any(t == r['task'] for t in tasks_included)]

sub_level_effect_size_df = convolve_sparse_scan_glm_with_cifti(parsed_valid_runs)
sub_level_effect_size_df.to_pickle('050923_effect_size_variance_df.pkl')