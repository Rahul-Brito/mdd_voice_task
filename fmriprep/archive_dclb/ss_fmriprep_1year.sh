#!/bin/bash

#SBATCH -p gablab
#SBATCH --time=6-23
#SBATCH --mem=45GB
#SBATCH --cpus-per-task=16
#SBATCH -J fmriprep
#SBATCH -o log/%x-%A-%a.out
#SBATCH -e log/%x-%A-%a.err
#SBATCH --mail-user=dclb@mit.edu
#SBATCH --mail-type=ALL

# grab these from submission script
bids_dir=$1
session=$2
args=($@)
subjs=(${args[@]:2}) #:2 needed because argument 1 is the input directory and argument 2 is the session, which we need to exclude

myscratch=/om2/scratch/Sat/$(whoami)

# set output for conversion
outdir=${myscratch}/mct/derivatives/ses-1year
export SINGULARITYENV_TEMPLATEFLOW_HOME=$outdir/.templateflow

# index slurm array to grab subject
subject=${subjs[${SLURM_ARRAY_TASK_ID}]}

echo Submitted subject: ${subject}

#set up singularity container
module add openmind/singularity/3.6.3
SING_IMG=/om2/user/dclb/containers/imaging/fmriprep_20.2.6.sif

# assign working directory
scratch=${myscratch}/fmriprep_work/${session}/${subject}
mkdir -p ${scratch}

#set fs license
export SINGULARITYENV_FS_LICENSE=/om2/user/dclb/imaging/fs_license_linux.txt

#set up
unset $PYTHONPATH
set -e

# default command
cmd="singularity run -e -B /om2 -B ${scratch}:/workdir -B ${bids_dir}:/input:ro -B ${outdir}:/out ${SING_IMG} /input /out participant --participant_label ${subject} -w /workdir --cifti-output 91k --mem_mb 40000 --output-spaces MNI152NLin6Asym:res-2 anat --fs-subjects-dir /out/freesurfer-${session}/ --bids-filter-file bids-filter-${session}.json --bids-database-dir ${myscratch}/mct/derivatives/bids_db/ --nprocs 16 --omp-nthreads 8"

printf "Command:\n${cmd}\n"

#remove an IsRunning files from freesurfer
rm -f ${outdir}/freesurfer-${session}/${subject}/scripts/IsRunning*

# run it
eval $cmd
