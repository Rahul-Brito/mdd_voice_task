import os

def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes

def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where

    allowed template fields - follow python string module:

    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    """
    rs=create_key('{bids_subject_session_dir}/func/sub-{subject}_{session}_task-rest_run-{item:01d}_bold')
    spin=create_key('{bids_subject_session_dir}/fmap/sub-{subject}_{session}_acq-func_dir-{dir}_run-{item:01d}_epi')
    dwi=create_key('{bids_subject_session_dir}/dwi/sub-{subject}_{session}_acq-{acq}_dwi')
    t1=create_key('{bids_subject_session_dir}/anat/sub-{subject}_{session}_run-{item:01d}_T1w')
    t2=create_key('{bids_subject_session_dir}/anat/sub-{subject}_{session}_T2w')
    pataka=create_key('{bids_subject_session_dir}/func/sub-{subject}_{session}_task-pataka_run-{item:01d}_bold')
    sentences=create_key('{bids_subject_session_dir}/func/sub-{subject}_{session}_task-pitchsent_run-{item:01d}_bold')
    nonwordrep=create_key('{bids_subject_session_dir}/func/sub-{subject}_{session}_task-nwr_run-{item:01d}_bold')
    facematch=create_key('{bids_subject_session_dir}/func/sub-{subject}_{session}_task-facematch_run-{item:01d}_bold')
    emosent=create_key('{bids_subject_session_dir}/func/sub-{subject}_{session}_task-emosent_run-{item:01d}_bold')
    vowels=create_key('{bids_subject_session_dir}/func/sub-{subject}_{session}_task-vowel_run-{item:01d}_bold')    
    pitch_emph=create_key('{bids_subject_session_dir}/func/sub-{subject}_{session}_task-pitchnw_run-{item:01d}_bold')
    movie_trailer=create_key('{bids_subject_session_dir}/func/sub-{subject}_{session}_task-movie_run-{item:01d}_bold')

    info = {rs: [], spin: [], dwi:[], t1:[], t2:[], 
            pataka:[], 
            sentences:[], 
            nonwordrep:[], 
            facematch:[], 
            emosent:[], 
            vowels:[], 
            pitch_emph:[], 
            movie_trailer:[]}

    for s in seqinfo:
        x,y,sl,nt = (s.dim1, s.dim2, s.dim3, s.dim4)
        if ('Movie' in s.protocol_name) and (sl == 65) and (nt > 69):
            info[movie_trailer].append(s[2])
        elif (sl == 65) and ('SMS5_rsfMRI' in s.protocol_name) and (nt > 150):
            if ('voice854' in s.patient_id):
                pass #add code later to account for rebooting during rsfc scan
            else:
                info[rs].append(s[2])
        elif (nt == 1) and ('Spin_Echo_EPI_' in s.protocol_name):
            if ('voice875' in s.patient_id) and ('4-Spin_Echo_EPI_PA' in s.series_id):
                pass
            elif ('voice_994' in s.patient_id) and ('5-Spin_Echo_EPI_AP' in s.series_id):
                pass
            else:
                dr = s.protocol_name.split('_')[-1]
                info[spin].append({'item': s[2], 'dir': dr})
        elif (nt == 72) and ('SMS2-diff' in s.protocol_name):
            if not s[13]:
                if 'PA' in s.protocol_name:
                    info[dwi].append({'item': s[2], 'acq': 'PA'})
        elif (nt == 7) and ('SMS2-diff' in s.protocol_name):
            if not s[13]:
                if 'AP' in s.protocol_name:
                    info[dwi].append({'item': s[2], 'acq': 'AP'})
        elif (sl == 176) and (nt == 1) and ('T1_MPRAGE' in s.protocol_name) and ('MEAN' in s.image_type):
            info[t1].append(s[2])
        elif ('T2_SPACE' in s.protocol_name) and ('NORM' in s.image_type):
            info[t2].append(s[2]) #=s.series_id
        elif ('PaTaKa' in s.protocol_name) and (sl == 45) and (nt > 103):
            info[pataka].append(s[2])
        elif ('Sentences' in s.protocol_name) and (sl == 65) and (nt >32):
            info[sentences].append(s[2])
        elif ('Nonword' in s.protocol_name) and (sl == 65) and (nt > 21):
            info[nonwordrep].append(s[2])
        elif ('FaceMatch' in s.protocol_name) and (nt > 49):
            info[facematch].append(s[2])
        elif ('EmoSent' in s.protocol_name) and (nt > 24):
            info[emosent].append(s[2])
        elif ('Vowels' in s.protocol_name) and (nt > 30):
            info[vowels].append(s[2])
        elif ('PitchEmph' in s.protocol_name) and (nt > 50):
            info[pitch_emph].append(s[2])
        else:
            pass
    for key, items in info.items():
        print(items)
    return info
