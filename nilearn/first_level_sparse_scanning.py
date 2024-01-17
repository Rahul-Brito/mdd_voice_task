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


def parse_valid_runs(tasks_included, events_path_pattern =  f'/nese/mit/group/sig/om_projects/voice/bids/data/sub-voice*/ses-*/func/*events.tsv', qc_filter = True, qc_tsv_file = '../../derivatives/first_level_110123/no_wall_of_speech_exclude_110123 - Sheet1.tsv'):
                     
    pop_events = find_populated_events(events_path_pattern)
    count_pop_events = pop_events.groupby('task').populated.value_counts()

    valid_runs = pop_events.events_file[pop_events.populated == True]
    parsed_valid_runs = [parse_file_entities(vr) for vr in valid_runs]

    subjects_excluded = ['voice997', 'voice897','voice863'] #for some reason these one is not in the fmriprep output
    parsed_valid_runs = [pvr for pvr in parsed_valid_runs if pvr['subject'] not in subjects_excluded]

    low_acompcor_to_drop = find_low_acompcor(parsed_valid_runs)
    parsed_valid_runs = [r for r in parsed_valid_runs if r not in low_acompcor_to_drop]

    parsed_valid_runs = [r for r in parsed_valid_runs if any(t == r['task'] for t in tasks_included)]
    [p.pop('datatype') for p in parsed_valid_runs]
    
    
    if qc_filter:
        #load list of runs to exclude and format as a list of bids-compliant names
        exclude_df = pd.read_table(qc_tsv_file)

        exclude = []
        for i, r in exclude_df.iterrows():
            sub = r['sub']
            ses = r['ses']
            task = r.iloc[5]
            run = r['run']
            exclude.append(parse_file_entities(f'/sub-voice{sub}_ses-{ses}_task-{task}_run-0{run}_events.tsv'))

        # remove excluded runs
        parsed_valid_runs = [p for p in parsed_valid_runs if p not in exclude]
    
    return parsed_valid_runs




def return_run_order(fmriprep_dir, tasks_included = ['nwr','pataka', 'emosent', 'vowel'], include_manual=True):
    #get all sub from fmriprep output
    subjects = [n.split('sub-')[1] for n in next(os.walk(fmriprep_dir))[1] if 'voice' in n]

    #986 and higher were control subjects so removing those
    #985 also seems like a control with lots of extra scans
    ctl_sub = ['voice'+ s for s in np.arange(985,1000).astype('str')]
    subjects = [s for s in subjects if s not in ctl_sub]

    tasks_included = tasks_included
    run_order = {}
    for s in subjects:
        base = f'/nese/mit/group/sig/om_projects/voice/rawData/{s}'
        for sd in next(os.walk(base))[1]:
            if 'behavioral' in os.listdir(base + '/'+ sd):
                for task in tasks_included:
                    #need to use sorted() so that glob organizes by the timestamp in the file
                    psychopy_logs = sorted(glob.glob(f'{base}/{sd}/behavioral/*{task}*.log'))
                    if s == 'voice980' and task == 'nwr': 
                        #for some reason the log file for this subject and task used X not R for run
                        run_numbers = [r.split('_X00')[1].split(f'_{task}')[0] for r in psychopy_logs]
                    else:
                        run_numbers = [r.split('_R00')[1].split(f'_{task}')[0] for r in psychopy_logs]
                    run_order[f'sub-{s}_task-{task}'] = run_numbers
    
    ### manual changes based on weird outputs and manual inspection
    ## change this flag to not include the manual fixes
    if include_manual:

        #original output says ['1', '1', '2', '2']. Inspecting log files at
        # ls /nese/mit/group/sig/om_projects/voice/rawData/voiceice844/session001_visit002/behavioral/*pataka*
        # and comparing to scan.tsv here /om2/scratch/Mon/rfbrito/bids/sub-voice844/ses-1/sub-voice844_ses-1_scans.tsv
        # shows the order of event files is 1 and 2, and the first two psychopy aren't used
        run_order['sub-voice844_task-pataka'] = ['1', '2'] 

        run_order['sub-voice854_task-nwr'] = ['1', '2'] #same thing for this situation
        #run_order['sub-voice856_task-emosent'] = ['1', '2']
        
        
        run_order['sub-voice859_task-emosent'] = ['1', '2'] #if the order is sorted already just drop the redundant ones

        ## this subject had no psychopy file for pataka, so i assume the order is right for the events for the 2 runs
        run_order['sub-voice860_task-pataka'] = ['1', '2'] 
        run_order['sub-voice860_task-emosent'] = ['1', '2']

        #blank becaus 884 has multiple behavioral folders
        #and my code went into session002_visit001/behavioral/ which didn't have pataka log files
        # even tho the subject did it
        run_order['sub-voice884_task-pataka'] = ['1', '2'] 

        ## 'sub-voice877_task-pataka' seemed to use the events for run1 for both run1 and run2

        #check e.g. /nese/mit/group/sig/om_projects/voice/rawData/voice889/session002_visit001/behavioral/*nwr*.psydat
        run_order['sub-voice889_task-nwr'] = ['1', '2', '3']

        #check e.g. /nese/mit/group/sig/om_projects/voice/rawData/voice889/session002_visit001/behavioral/*pataka*.psydat
        run_order['sub-voice889_task-pataka'] = ['1', '2'] 

        #same for ses1 and ses1
        run_order['sub-voice893_task-pataka'] = ['1', '2'] 

        #'sub-voice897_task-pataka' seems to use same events for run1 and run 2
        run_order['sub-voice953_task-pataka'] = ['1', '2'] 
        
        #same for ses-1 and ses-2
        run_order['sub-voice956_task-pataka'] = ['1', '2'] 
        
        run_order['sub-voice957_task-pataka'] = ['1', '2'] 
        
        #not sure why this came out ['1', '3', '3', '2'] but check 
        # /nese/mit/group/sig/om_projects/voice/rawData/voice958/session001_visit002/behavioral/*nwr*psydat
        run_order['sub-voice958_task-nwr'] = ['1', '2', '3']
        
        #drop redundant run1
        run_order['sub-voice962_task-pataka'] = ['1', '2'] 
        
        #sub-voice964 had a second session with other tasks but not the main ones
        run_order['sub-voice964_task-nwr'] = ['1', '2', '3']
        run_order['sub-voice964_task-pataka'] = ['1', '2']
        run_order['sub-voice964_task-emosent'] = ['1', '2']
        run_order['sub-voice964_task-vowel'] = ['1', '2']
        
        #extra run 1
        run_order['sub-voice967_task-pataka'] = ['1', '2']
        #967 did not do nwr, checked scans.tsv
        
        #was a run 1 and run 3 no run 2, unsure what was being modele for run3 so dropping
        run_order['sub-voice968_task-vowel'] = ['1']
        
        #973 did not do nwr or pataka, checked scans.tsv and imaging outputs
        
        #975 used conditions/pataka_run1.xlsx for both runs
        
        #has run 1 and 2 in session001_visit001/behavioral/*nwr*
        #has run 2 and 3 in session001_visit002/behavioral/*nwr*
        #scans.tsv shows heudiconv numbering of run3, 2, 1
        #checking timestamps and log files, it was just 1, 2, 3 that was done in that order and kept
        run_order['sub-voice979_task-nwr'] = ['1', '2', '3']
        
        #Assuming 979 did pataka in order 1,2. No psychopy file
        run_order['sub-voice979_task-pataka'] = ['1', '2']
        
        #Assuming 980 did pataka in order 1,2. No psychopy file
        run_order['sub-voice980_task-pataka'] = ['1', '2']
        
        #Assuming 981 did pataka in order 1,2. No psychopy file
        run_order['sub-voice980_task-pataka'] = ['1', '2']
        
        #982 did not do nwr
        #Assuming 982 did pataka and vowel in order 1,2. No psychopy file
        run_order['sub-voice982_task-pataka'] = ['1', '2']
        run_order['sub-voice982_task-vowel'] = ['1', '2']
        
        #983 did not do nwr
        #Assuming 983 did pataka in order 1,2. No psychopy file
        run_order['sub-voice983_task-pataka'] = ['1', '2']
        
        #Assuming 983 did pataka in order 1,2. No psychopy file
        run_order['sub-voice984_task-pataka'] = ['1', '2']
        
        #kept in case i bring 985 back
        #from session001_visit001/behavioral/ which matches the time stamps in the scans.tsv it was 1,2
        #run_order['sub-voice985_task-pataka'] = ['1', '2']
        #from session001_visit001/behavioral/ which matches the time stamps in the scans.tsv it was 1,2,3
        #run_order['sub-voice985_task-nwr'] = ['1', '2', '3']
        
        
    return run_order





def create_contrast(design_matrix, task, ohbm, con):
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

        if con == 'happyvneutral':
            speech_contrast = conditions['happy'] - conditions['neutral'] 
        elif con == 'sadvneutral':
            speech_contrast = conditions['sad'] - conditions['neutral']
        elif con == 'emoftest':
            speech_contrast = [conditions['happy'] - conditions['neutral'],
                               conditions['sad'] - conditions['neutral']]
#                 speech_contrast = [conditions['happy'] - conditions['silent'],
#                                    conditions['sad'] - conditions['silent'],
#                                    conditions['neutral'] - conditions['silent']]
        elif con == 'happyvsil':
            speech_contrast = conditions['happy'] - conditions['silent']
        elif con == 'neutralvsil':
            speech_contrast = conditions['neutral'] - conditions['silent']
        elif con == 'sadvsil':
            speech_contrast = conditions['sad'] - conditions['silent']
        elif con == 'speechvsil':
            speech_contrast = 0.33*conditions['happy'] + 0.33*conditions['neutral'] + 0.33*conditions['sad'] - conditions['silent']

    if task == 'nwr':
        if ohbm:
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
            
            #to try to fit a linear line between these, we don't give a contrast value to baseline
            speech_contrast = -1*conditions['2'] - 0.33*conditions['3'] + 0.33*conditions['4'] + 1*conditions['5']
        else:
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







def smooth_cifti(fmriprep_dir,sub,task,ses,run, resmooth):
    #L-R surface templates
    left_surface = '/om2/user/jsmentch/data/datalad/templateflow/tpl-fsLR/tpl-fsLR_hemi-L_den-32k_sphere.surf.gii'
    right_surface = '/om2/user/jsmentch/data/datalad/templateflow/tpl-fsLR/tpl-fsLR_hemi-R_den-32k_sphere.surf.gii'
    
    ###Load CIFTI, smooth, and save
    
    #fmriprep dir for each subject
    ses_dir = f'{fmriprep_dir}/sub-{sub}/ses-{ses}'

    # add smoothed and cleaned dir to fmriprep for each sub
    smoothed_dir = f'{ses_dir}/smoothed_110123'
    #cleaned_dir = f'{ses_dir}/cleaned'
    
    #get the cifti file
    func_file = glob.glob(f'{fmriprep_dir}/sub-{sub}/ses-{ses}/func/sub-{sub}*ses-{ses}*task-{task}*run-{run}*fsLR_den-91k_bold.dtseries.nii')[0]  
    
    #create name of smoothed file to save to
    smooth_output_file = f'{smoothed_dir}/{os.path.basename(func_file)}'

    #if we want to resmooth we resmooth and write to smooth_output file
    if resmooth:
        #create directories for smoothed and cleaned data
        os.makedirs(smoothed_dir, exist_ok=True)
        #os.makedirs(cleaned_dir, exist_ok=True)

        #smoothing cifti files using connectome workbench
        
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
    else:
        #if we dont want to resmooth we use the same smoothed file
        smoothed_func_img = nimg.load_img(smooth_output_file)
        smoothed_func_signal = smoothed_func_img.get_fdata()
    
    return smoothed_func_signal, func_file





def generate_sparse_scan_regressors(nifti, frame_times, task_json, events, task):
    
    TR=task_json['RepetitionTime']
    DT=task_json['DelayTime']

    sparse_model = modelgen.SpecifySparseModel()
    sparse_model.inputs.input_units = 'secs'
    sparse_model.inputs.functional_runs = nifti
    sparse_model.inputs.time_repetition = TR #sets TR to the full window of MRI acq and bold signal [silent+speak]
    sparse_model.inputs.time_acquisition = TR - DT # specifies MR acquisition time
    
    
    if task == 'nwr':
        sparse_model.inputs.scan_onset = -(TR - DT) #specifies onset since acqusition happens first (is this right)
    elif task == 'emosent':
        sparse_model.inputs.scan_onset = 0
    
    #sparse_model.inputs.high_pass_filter_cutoff = 128. #satra said to remove
    sparse_model.inputs.model_hrf = True
    sparse_model.inputs.subject_info = modelgen.bids_gen_info(events,condition_column='trial_type')  # doctest: +SKIP

    #potentially could also tack on the noise regressors too here
    regressors = sparse_model._list_outputs()['session_info'][0]['regress']
    data = [v['val'] for v in regressors]
    col = [t['name'] for t in regressors]
    df_regressors = pd.DataFrame(data).T
    df_regressors.columns = col
    df_regressors.index = frame_times
    
    return df_regressors






def save_cifti(cifti_data, output_type, out_dir, sstr, con, sample_cifti_file='../../derivatives/fmriprep/sub-voice857/ses-1/func/sub-voice857_ses-1_task-emosent_run-2_space-fsLR_den-91k_bold.dtseries.nii'):
    #takes 91k np array of parcels and generates cifti object
    
    sample_cifti = nimg.load_img(sample_cifti_file)
    time_axis, brain_model_axis = [sample_cifti.header.get_axis(i) for i in range(sample_cifti.ndim)]
    z = np.atleast_2d(cifti_data)
    scalar_axis = nib.cifti2.ScalarAxis([output_type])  # Takes a list of names, one per row
    new_header = nib.Cifti2Header.from_axes([scalar_axis, brain_model_axis])
    #z = np.reshape(z, (-1, z.shape[0]))
    img = nib.Cifti2Image(z, new_header)

    name = f'{out_dir}/{sstr}_space-fsLR_den-91k_contrast-{con}_{output_type}.dscalar.nii'
    img.to_filename(name)
    return name






def save_first_level_output(contrast_output, out_dir, sstr, con):    
    save_cifti(contrast_output.effect_size(), 'effect_size', out_dir, sstr, con)
    save_cifti(contrast_output.z_score(), 'z_score', out_dir, sstr, con)
    save_cifti(contrast_output.effect_variance(), 'effect_variance', out_dir, sstr, con)
          
    print(f'finished {sstr}')
    
    return

    
    
    
    

def convolve_sparse_scan_glm_with_cifti(parsed_valid_runs, out_dir, ohbm, con, resmooth, runshift):
    #TO DO: Docstring on what this does, inputs, outputs, types (including within the dictionary) (pvr - union)
    #TO DO: use pybids objects
    
    #TO DO: make an input parameter
    #base directory for fmriprep output
    fmriprep_dir = '../../derivatives/fmriprep'

    
    #query list of subjects and runs
    # subjects = layout.get_subjects()
    # runs = layout.get_runs()

    sparse = True
    space='MNI152NLin6Asym'

    first_level_stats_maps = {}
    
    fail_log = []
    
    #TO DO: Use python logger
    probe = [] #to append and print any probes
    
    #TO DO: make it run on subject, have a job to run subjects/task/run in parallel 
    
    #TO DO: check if files exist that are needed?
    
    for pvr in parsed_valid_runs:
        
        #TO DO: spell out subject, session etc. so the code is readable
        sub = pvr['subject']
        ses = pvr['session']
        run = int(pvr['run']) #TO DO: comment why I cast it. Maybe leave it as a string.
        task = pvr['task']
        
        print(f'started sub-{sub}_ses-{ses}_task-{task}_rec-unco_run-{run}')

        ### Load CIFTI, smooth, and save
        ## if already smoothed then reload previously smoothed file
        
        #TO DO: add memoization (?) so code checks to see if something exists and doesn't redo it
        smoothed_func_signal, the_input = smooth_cifti(fmriprep_dir,sub,task,ses,run,resmooth)

        ### scale the signal
        #TO DO: don't necessarily hard-code the 10000, am I sure it's right, etc.
        #TO DO: maybe add a small epsilon to the denominator so i don't get nans?
        smoothed_func_signal = 10000 * np.divide(smoothed_func_signal - np.min(smoothed_func_signal),
                                                 np.mean(smoothed_func_signal - np.min(smoothed_func_signal)))
        
        ### Get spare resampled timestamps from volumetric data
        
        

        # Need nifti template for nipype sparse scan convolution for this specific sub/ses/task/run
        nifti = glob.glob(f'../../derivatives/fmriprep/sub-{sub}/ses-{ses}/func/sub-{sub}*task-{task}*run-{run}*{space}*preproc*nii.gz')[0]
        
        #fix run order to pull right events file based on faulty dicom to nifti mapping
        #this uses psychopy log files for each subject/task
        #run_order = return_run_order(fmriprep_dir, tasks_included = ['nwr','pataka', 'emosent', 'vowel'], include_manual=True)
        
        
#         if runshift:
#             run_event = run_order[f'sub-{sub}_task-{task}'][run-2]
#         else:
#             run_event = run_order[f'sub-{sub}_task-{task}'][run-1]
#         events = glob.glob(f'/nese/mit/group/sig/om_projects/voice/bids/data/sub-{sub}/ses-{ses}/func/sub-{sub}*task-{task}*run-0{run_event}*events.tsv')
        
        #TO DO: delete later
        if task == 'emosent':
            events = [f'test/sub-{sub}_ses-{ses}_task-{task}_run-{run}_events.tsv']
            #event_data = pd.read_table(events[0])
            #event_data.duration = 4
            #events = [f'../../derivatives/events_emosent_112823/sub-{sub}_ses-{ses}_task-{task}_run-0{run_event}_events.tsv']
            #event_data.to_csv(events[0], sep="\t", index=False)
           
        #get the confounds to regress against for 1st level model

        selected_confounds=get_confounds(sub,task,ses,run)
            
        #generate frame times for design matrix
#         nscans = smoothed_func_signal.shape[0]
#         start_time = 0 * TR
#         end_time = ((nscans - 1) *TR)
#         frame_times = np.linspace(start_time, end_time, nscans)
#         frame_times
        
        #load task JSON to get TR 
        #TO DO: Pybids for loading these files so it can handle the hierarchical nature of bids files
        #TO DO: add 'with open" to make sure file is closed after using it
        #TO DO: add "/r" for read-only to openning the file so I don't accidentally change it
        #TO DO: implement YODA for dir organizatin (https://handbook.datalad.org/en/latest/basics/101-127-yoda.html)
        task_json = open(f"../../task-{task}_bold.json")
        task_json=json.load(task_json)
        TR=task_json['RepetitionTime']
        
        #get frame times from events file
        frame_times = pd.read_table(events[0]).onset.to_numpy()
        frame_times = np.append(frame_times, frame_times[-1] + TR) #events file from psychopy is always a row short?
        
        #create resampled regressors for each task condition
        try:
            # outputs regressors for each task condition. Convolves with HRF and resamples based on sparse scan timing. 
            #Requires the original nifti file
            sparse_scan_regressors = generate_sparse_scan_regressors(nifti, frame_times, task_json, events, task)
            fails = None
        except Exception as Arguement: #TO DO: try to catch the specifc error e.g. except key_error or something
            fails = Arguement
            fail_log.append((pvr, fails))
            continue
        sparse_scan_regressors = sparse_scan_regressors.assign(silent = 0)

        #TO DO: add more helper functions
        
        #create design matrix with sparse scan regressors, noise regressors, and a column of 1s for the intercept
        #1for contrast, order is still [sparse task regressors, noise regressors, intercept]
        selected_confounds.index = sparse_scan_regressors.index
        intercept = pd.Series(np.ones(sparse_scan_regressors.shape[0]), name='intercept',
                              index=sparse_scan_regressors.index) #col of 1s
        design_matrix = pd.concat([sparse_scan_regressors, selected_confounds, intercept], axis=1)
        probe.append(design_matrix)
        
        #run the glm, with ar1 noise model
        labels, estimates = run_glm(smoothed_func_signal, design_matrix.values, noise_model='ar1', n_jobs=-2)
        
        #create the task-specific contrast
        speech_contrasts = create_contrast(design_matrix, task, ohbm, con)
        
        #compute contrast output (nilearn object with effect size, z-stat, etc. for this contrast
        #it should pick the right contrast type (t or F) depending on 1 or multi-row design contrast
        contrast_output = compute_contrast(labels, estimates, speech_contrasts)#,
                                       # contrast_type='t')
        
        save_first_level_output(contrast_output, out_dir, f'sub-{sub}_ses-{ses}_task-{task}_rec-unco_run-{run}', con)
        
        #TO DO: flake-8 or ruff, black (wraps flake-8) etc. for code formatting? But work w python files not with notebooks
        
        #probe.append([the_input, events])
        

        #return the output type we want
#         if return_type == 'effect_size':
#             first_level_stats_maps[f'sub-{sub}_ses-{ses}_task-{task}_run-{run}'] = contrast_output.effect_size()
#         elif return_type == 'z_score':
#             first_level_stats_maps[f'sub-{sub}_ses-{ses}_task-{task}_run-{run}'] = contrast_output.z_score()
#         elif return_type == 'effect_variance':
#             first_level_stats_maps[f'sub-{sub}_ses-{ses}_task-{task}_run-{run}'] = contrast_output.effect_variance()

#     first_level_stats_maps_df = pd.DataFrame(first_level_stats_maps)
    with open(f'{out_dir}/fails.txt', 'w') as f:
        for line in fail_log:
            f.write(f"{line}\n")
    
    #print(speech_contrasts, design_matrix.columns)
    return probe #selected_confounds #events#speech_contrasts, design_matrix#.columns #events, , design_matrix



