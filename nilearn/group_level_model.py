#Import packages
import os
import glob
import json
import pickle
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
from nilearn.glm.contrasts import compute_contrast, compute_fixed_effects

from bids.layout import BIDSLayout, parse_file_entities

# import cortex
# from cortex import fmriprep

from nipype.interfaces.workbench.base import WBCommand
from nipype.algorithms import modelgen
from nipype.interfaces.base import Bunch

import scipy.stats as stats

from statsmodels.api import OLS

import hcp_utils as hcp

import utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import importlib

#! module load openmind/hcp-workbench/1.2.3

import numpy as np
import pandas as pd

def load_phenotypic_data(filename, demean):        
    #loading my covariates and putting into the second level design matrix
    pheno = pd.read_csv(filename, index_col=0)[['voice_id','beckdepressionii_total','age', 'sex']].reset_index(drop=True)

    #955 is a duplicate but from the audio i determined it is a young female not an old man
    #removing the old man from the phenotype list
    pheno = pheno[~((pheno.voice_id == 955) & (pheno.age == 53) & (pheno.sex == 1))]

    #add MDD label for ease of data sorting
    pheno['mdd'] = pheno.beckdepressionii_total >= 14 #(pheno.beckdepressionii_total >= 14)
    pheno.mdd = pheno.mdd.replace({True:'mdd', False:'control'})

    if demean:
        pheno.age = pheno.age - pheno.age.mean() # so I demeaned everything for now...
        pheno.beckdepressionii_total = pheno.beckdepressionii_total - pheno.beckdepressionii_total.mean()
    
    return pheno





def create_task_design_matrix(second_level_effects, pheno):
    
    task_design_matrix = {}
    for task, fx in second_level_effects.items():
        pheno_per_ses = []
        for file in fx:
            info = parse_file_entities(file)
            sub = int(info['subject'].split('voice')[1])
            demo = pheno[pheno.voice_id == sub]
            #demo.loc[:,'voice_id'] = str(sub) + '_ses-' + info['session']
            demo.loc[:,'voice_id'] = str(sub)
            pheno_per_ses.append(demo)

        design_matrix = pd.concat(pheno_per_ses)
        design_matrix.index = design_matrix.voice_id
        design_matrix = design_matrix.drop('voice_id', axis=1)
        design_matrix['intercept'] = np.ones(design_matrix.shape[0])
        #design_matrix = design_matrix[covariate_list]
        task_design_matrix[task] = design_matrix
    return task_design_matrix






def create_group_contrast(design_matrix, covariate):
    contrast = np.zeros(design_matrix.shape[1])
    cov_index = design_matrix.columns.get_loc(covariate)
    contrast[cov_index] = 1
    return contrast






def fit_group_model(effect_size_signals_combined, task_design_matrix):
    model_out={}

    #for sc, cov_design_matrix in sub_cov_design_matrix.items():
    #fx_sig = effect_size_signals_combined[sc]

    groups = ['mdd','all_sub'] #mdd, all subjects

    for task,betas in effect_size_signals_combined.items():
        #dm = task_design_matrix[task]
        #pd.concat([cov_design_matrix['bdi'], regressors[condition]], axis=1) #add covariates

        #get design matrix either for just subjects above BDI cutoff or all subjects
        contrast_out = {}
        for grp in groups:
            if grp == 'mdd':
                design_matrix = task_design_matrix[task][task_design_matrix[task].mdd == grp].drop('mdd', axis=1)
                
                betas_mdd =  betas[betas.index.isin(design_matrix.index)]

                labels, estimates = run_glm(betas_mdd.values, design_matrix.values, noise_model='ols', 
                                        n_jobs=-2, verbose=0)

                #look at specific contrast
                cov = 'beckdepressionii_total'
                contrast = create_group_contrast(design_matrix, cov)
                contrast_out[grp] = compute_contrast(labels, estimates, contrast)
            else:
                design_matrix = task_design_matrix[task].drop('mdd', axis=1)

                labels, estimates = run_glm(betas.values, design_matrix.values, noise_model='ols', 
                                        n_jobs=-2, verbose=0)

                #look at specific contrast
                cov = 'beckdepressionii_total'
                contrast = group_level_model.create_group_contrast(design_matrix, cov)
                contrast_out[grp] = compute_contrast(labels, estimates, contrast)
        model_out[task] = contrast_out
    return model_out
