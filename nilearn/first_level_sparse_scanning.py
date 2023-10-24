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
from sklearn.linear_model import LinearRegression
from scipy import stats
import statsmodels.api as sm

#%matplotlib inline
#! module load openmind/hcp-workbench/1.2.3

def find_populated_events(events_path_pattern):

    #get list of all events file in bids directory (using old location) per task as df
    #pop_events = pd.DataFrame(glob.glob(f'/om/project/voice/bids/data/sub-voice*/ses-*/func/*{task}*events.tsv'), columns= ['events_file'])

    pop_events = pd.DataFrame(glob.glob(events_path_pattern), columns= ['events_file'])

    #use pd empty attribute to determine if file is empty or not. 
    #invert the boolean value as I want to know if it is popualated or not
    pop_events['populated'] = [not pd.read_table(f).empty for f in pop_events.events_file]

    pop_events['task'] = [parse_file_entities(file)['task'] for file in pop_events.events_file]

    return pop_events

def create_contrast(design_matrix, task):
    #fmri_img = concat_imgs(nifti)
    #mean_img = mean_img(fmri_img)
    
    speech_contrast = np.zeros(design_matrix.shape[1])
    
    if task == 'pataka':
        clear = np.zeros(design_matrix.shape[1])
        clear[0] = 1
        normal = np.zeros(design_matrix.shape[1])
        normal[1] = 1
        rapid = np.zeros(design_matrix.shape[1])
        rapid[2] = 1
        silent = np.zeros(design_matrix.shape[1])
        silent[3] = 1


        conditions = {
            'clear': clear,
            'normal': normal,
            'rapid': rapid,
            'silent': silent
        }

        speech_contrast = 0.33*conditions['clear'] + 0.33*conditions['normal'] + 0.33*conditions['rapid'] - conditions['silent']
    
    if task == 'emosent':
        happy= np.zeros(design_matrix.shape[1])
        happy[0] = 1
        neutral= np.zeros(design_matrix.shape[1])
        neutral[1] = 1
        sad = np.zeros(design_matrix.shape[1])
        sad[2] = 1
        silent = np.zeros(design_matrix.shape[1])
        silent[3] = 1
        
        conditions = {
            'happy': happy,
            'neutral': neutral,
            'sad': sad,
            'silent': silent
        }
        
        speech_contrast = 0.33*conditions['happy'] + 0.33*conditions['neutral'] + 0.33*conditions['sad'] - conditions['silent']
    
    if task == 'nwr':
        two = np.zeros(design_matrix.shape[1])
        two[0] = 1
        three= np.zeros(design_matrix.shape[1])
        three[1] = 1
        four = np.zeros(design_matrix.shape[1])
        four[2] = 1
        five = np.zeros(design_matrix.shape[1])
        five[3] = 1
        rest = np.zeros(design_matrix.shape[1])
        rest[4] = 1
        
        conditions = {
            '2': two,
            '3': three,
            '4': four,
            '5': five,
            'Rest' : rest
        }
        
        speech_contrast = 0.25*conditions['2'] + 0.25*conditions['3'] + 0.25*conditions['4'] + 0.25*conditions['5'] - conditions['Rest']
    
    if task == 'vowel':
        
        high = np.zeros(design_matrix.shape[1])
        high[0] = 1
        low= np.zeros(design_matrix.shape[1])
        low[1] = 1
        normal = np.zeros(design_matrix.shape[1])
        normal[2] = 1
        silent = np.zeros(design_matrix.shape[1])
        silent[3] = 1
        
        conditions = {
            'high': high,
            'low': low,
            'normal': normal,
            'silent': silent
        }
        
        speech_contrast = 0.33*conditions['high'] + 0.33*conditions['low'] + 0.33*conditions['normal'] - conditions['silent']
    
    return speech_contrast

def find_low_acompcor(parsed_valid_runs):
    low_acompcor_to_drop = []

    for pvr in parsed_valid_runs:
        sub = pvr['subject']
        ses = pvr['session']
        run = int(pvr['run'])
        task = pvr['task']

        all_confounds = pd.read_csv(f"../../derivatives/fmriprep/sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-{task}_run-{run}_desc-confounds_timeseries.tsv", sep = '\t')

        all_confounds_json = open(f"../../derivatives/fmriprep/sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-{task}_run-{run}_desc-confounds_timeseries.json")
        all_confounds_json=json.load(all_confounds_json)

        #num_a_comp_cors[f'sub-{sub}_ses-{ses}_task-{task}_run-{run}'] = (len([col for col in all_confounds.columns if 'a_comp_cor' in col]))

        if (len([col for col in all_confounds.columns if 'a_comp_cor' in col])) < 5:
            low_acompcor_to_drop.append(pvr)
            
    return low_acompcor_to_drop


def get_confounds(sub,task,ses,run):

    all_confounds = pd.read_csv(f"../../derivatives/fmriprep/sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-{task}_run-{run}_desc-confounds_timeseries.tsv", sep = '\t')
    
    all_confounds_json = open(f"../../derivatives/fmriprep/sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-{task}_run-{run}_desc-confounds_timeseries.json")
    all_confounds_json=json.load(all_confounds_json)
    
    
    #typically included rigid body motion (or those plus squares and derivatives if desired, then must comment out top line and uncomment bottom 3)
    motion_params = ['trans_x', 'trans_y', 'trans_z','rot_x','rot_y','rot_z']
    #motion_trans_params = [col for col in all_confounds.columns if 'trans' in col] #change to these squares and derivatives if desired
    #motion_rot_params = [col for col in all_confounds.columns if 'rot' in col] #change to these if desired
    #motion_params=motion_trans_params+motion_rot_params #change to these if desired

    
    #individual col with single 1 for timepoint of motion
    #motion_outliers = [col for col in all_confounds.columns if 'motion_outlier' in col]  
    
    
    #for low freq signal drift
    #cannot include this and high-pass temp filter bc already removes low freq fluc
    #required if using aCompCor (or tCompCor)
    cosine_regressors = [col for col in all_confounds.columns if 'cosine' in col] 
    
    
    #these can be adjusted to be from the combined wm csf, for example
    #doesn't make sense to use csf and wm signal regression if using these according to fmriprep documentation
    #6 is rule of thumb; can pick diff number or specific amount of variance explained
    #TO DO clarify if edge/crown regressors are already part of compcor -- unclear in docs and can't find separate regressor in tsv
    num_a_comp_cors=5
    a_comp_cors = []
    for i in range(num_a_comp_cors):
        a_comp_cors.append('a_comp_cor_{:02d}'.format(i))
    
        
    #if taking ICA AROMA denoised niftis (~desc-smoothAROMAnonaggr_bold.nii.gz), can't also include ICA noise regressors & MUST drop non-steady state vols
    #here we are taking instead the ICA AROMA regressors: aroma_motion_XX
#     aroma_regressors_all = [col for col in all_confounds.columns if 'aroma' in col]
#     aroma_regressors_noise=[]
#     #TO DO: excluding for now, but double check on this!
#     for regr in aroma_regressors_all:
#         json_name ='aroma_motion_'+str(int(regr.split('aroma_motion_')[1]))
#         if all_confounds_json[json_name]['MotionNoise']==True:
#             aroma_regressors_noise.append(regr)
        

    #we need to filter out non-steady state volumes if using cosine regressors, ICA AROMA and CompCor regressors...    
    non_steady_state_regressors = [col for col in all_confounds.columns if 'non_steady_state' in col]
           
    #TO DO: not sure if CSF should be kept since already have aCompCors (excluding for now)
    #selected_confounds = all_confounds[['framewise_displacement']+motion_params+motion_outliers+cosine_regressors+a_comp_cors+aroma_regressors_noise+non_steady_state_regressors].copy()

    #selected_confounds = all_confounds[['framewise_displacement']+motion_params+cosine_regressors+a_comp_cors+aroma_regressors_noise+non_steady_state_regressors].copy()

    selected_confounds = all_confounds[['framewise_displacement']+motion_params+cosine_regressors+a_comp_cors+non_steady_state_regressors].copy()

    #selected_confounds = all_confounds[['framewise_displacement']+motion_params].copy()
    
    #selected_confounds = all_confounds[['framewise_displacement']+motion_params+motion_outliers].copy()

    #get rid of nas in first row of derivative and framewise displacement cols
    for col in selected_confounds.columns:
        if ('derivative' in col) or ('framewise_displacement' in col):
            if pd.isna(selected_confounds[col][0]):
                selected_confounds[col][0]=0

    return selected_confounds


def smooth_cifti(fmriprep_dir,sub,task,ses,run):
    #L-R surface templates
    left_surface = '/om2/user/jsmentch/data/datalad/templateflow/tpl-fsLR/tpl-fsLR_hemi-L_den-32k_sphere.surf.gii'
    right_surface = '/om2/user/jsmentch/data/datalad/templateflow/tpl-fsLR/tpl-fsLR_hemi-R_den-32k_sphere.surf.gii'
    
    ###Load CIFTI, smooth, and save
    
    #fmriprep dir for each subject
    ses_dir = f'{fmriprep_dir}/sub-{sub}/ses-{ses}'

    # add smoothed and cleaned dir to fmriprep for each sub
    smoothed_dir = f'{ses_dir}/smoothed'
    #cleaned_dir = f'{ses_dir}/cleaned'

    #create directories for smoothed and cleaned data
    os.makedirs(smoothed_dir, exist_ok=True)
    #os.makedirs(cleaned_dir, exist_ok=True)    

    #get the cifti file
    func_file = glob.glob(f'{fmriprep_dir}/sub-{sub}/ses-{ses}/func/sub-{sub}*ses-{ses}*task-{task}*run-{run}*fsLR_den-91k_bold.dtseries.nii')[0]                      

    #smoothing cifti files using connectome workbench
    smooth_output_file = f'{smoothed_dir}/{os.path.basename(func_file)}'
    wb_command = WBCommand(command='wb_command')
    
    # smooth with 4mm FWHM kernel in volume and surface. Note -fwhm flag now
    wb_command.inputs.args = f'-cifti-smoothing {func_file} 4 4 COLUMN {smooth_output_file} -fwhm -right-surface {right_surface} -left-surface {left_surface}'
    wb_command.run()

    #load smoothed func data
    smoothed_func_img = nimg.load_img(smooth_output_file)
    smoothed_func_signal = smoothed_func_img.get_fdata()

    #give it the right header it seems
    func_smooth = nib.Cifti2Image(smoothed_func_signal, smoothed_func_img.header)

    #save the file with the right header as the final output
    smooth_output_file = f'{smoothed_dir}/{os.path.basename(func_file)}'
    func_smooth.to_filename(smooth_output_file)
    
    return smoothed_func_signal

def generate_sparse_scan_regressors(nifti, frame_times, task_json, events):
    
    TR=task_json['RepetitionTime']
    DT=task_json['DelayTime']

    sparse_model = modelgen.SpecifySparseModel()
    sparse_model.inputs.input_units = 'secs'
    sparse_model.inputs.functional_runs = nifti
    sparse_model.inputs.time_repetition = TR
    sparse_model.inputs.time_acquisition = TR - DT
    sparse_model.inputs.high_pass_filter_cutoff = 128.
    sparse_model.inputs.model_hrf = True
    sparse_model.inputs.subject_info = modelgen.bids_gen_info(events,condition_column='trial_type')  # doctest: +SKIP

    regressors = sparse_model._list_outputs()['session_info'][0]['regress']
    data = [v['val'] for v in regressors]
    col = [t['name'] for t in regressors]
    df_regressors = pd.DataFrame(data).T
    df_regressors.columns = col
    df_regressors.index = frame_times
    
    return df_regressors

def convolve_sparse_scan_glm_with_cifti(parsed_valid_runs, return_type, custom_events = []):
    #base directory for fmriprep output
    fmriprep_dir = '../../derivatives/fmriprep'

    
    #query list of subjects and runs
    # subjects = layout.get_subjects()
    # runs = layout.get_runs()

    #for PVR in PARSED VALID RUNS of pataka only

    sparse = True
    space='MNI152NLin6Asym'

    first_level_stats_maps = {}

    for pvr in parsed_valid_runs:
        sub = pvr['subject']
        ses = pvr['session']
        run = int(pvr['run'])
        task = pvr['task']

        ### Load CIFTI, smooth, and save
        smoothed_func_signal = smooth_cifti(fmriprep_dir,sub,task,ses,run)

        ##### We are here now
        
        ### Get spare resampled timestamps from volumetric data
        
        #load task JSON to get TR 
        task_json = open(f"../../task-{task}_bold.json")
        task_json=json.load(task_json)
        TR=task_json['RepetitionTime']

        # Need nifti template for nipype sparse scan convolution for this specific sub/ses/task/run
        nifti = glob.glob(f'../../derivatives/fmriprep/sub-{sub}/ses-{ses}/func/sub-{sub}*task-{task}*run-{run}*{space}*preproc*nii.gz')[0]
        
        if not custom_events:
            events = glob.glob(f'/nese/mit/group/sig/om_projects/voice/bids/data/sub-{sub}/ses-{ses}/func/sub-{sub}*task-{task}*run-0{run}*events.tsv')
        else:
            events = custom_events

        #get the confounds to regress against for 1st level model
        selected_confounds=get_confounds(sub,task,ses,run)

        #generate frame times for design matrix
        nscans = smoothed_func_signal.shape[0]
        start_time = 0 * TR
        end_time = ((nscans - 1) *TR)
        frame_times = np.linspace(start_time, end_time, nscans)
        frame_times
        

        #create resampled regressors for each task condition
        if sparse:
            try:
                # outputs regressors for each task condition. Requires the original nifti file
                sparse_scan_regressors = generate_sparse_scan_regressors(nifti, frame_times, task_json, events)
                fails = None
            except Exception as Arguement:
                fails = Arguement
            
        #create design matrix with resampled task regressors and confounds
        selected_confounds.index = sparse_scan_regressors.index
        design_matrix = pd.concat([sparse_scan_regressors, selected_confounds], axis=1)

        #create the task-specific contrast
        speech_contrasts = create_contrast(design_matrix, task)
        
        #did manual first level model 
        # y = betas*x
        # y = the smoothed cifti image for this subject task/run
        # x = the design matrix with the resampled task regressors and the confounds
        # betas = the betas we fit for each greyordinate
        
        
        ##### MISSING NOISE MODEL???
        
        #need to add the intercept column of 1s (called "const" for statsmodel)
        design_matrix = sm.add_constant(design_matrix)
        first_level_model = sm.OLS(smoothed_func_signal, design_matrix) 
        fit_glm = first_level_model.fit()
        
        #matrix multiply by the desired contrast for the task, to isolate the 
        #speech >silent contrast while controlling for the regressors
        
        betas = fit_glm.params.T.drop('const', axis=1)#get betas from the model, drop intercept column
        contrast_map = np.matmul(betas,speech_contrasts)
        
        #return the output type we want
        if return_type == 'effect_size':
            first_level_stats_maps[f'sub-{sub}_ses-{ses}_task-{task}_run-{run}'] = contrast_map
        elif return_type == 'z_score':
            first_level_stats_maps[f'sub-{sub}_ses-{ses}_task-{task}_run-{run}'] = stats.zscore(contrast_map)

    first_level_stats_maps_df = pd.DataFrame(first_level_stats_maps)
    return fit_glm
    #return first_level_stats_maps_df 