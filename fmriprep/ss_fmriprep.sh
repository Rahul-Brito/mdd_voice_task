#!/bin/bash

#SBATCH -p gablab
#SBATCH --time=6-23
#SBATCH --mem=45GB
#SBATCH --cpus-per-task=16
#SBATCH -J fmriprep
#SBATCH -o log/%x-%A-%a.out
#SBATCH -e log/%x-%A-%a.err
#SBATCH --mail-user=rfbrito@mit.edu
#SBATCH --mail-type=ALL

# grab these from submission script
bids_dir=$1
args=($@)
#subjs=(${args[@]:1}) #:2 needed because argument 1 is the input directory and argument 2 is the session, which we need to exclude
subject=$2

myscratch=/om2/scratch/Wed/rfbrito

# set output for conversion
outdir=${myscratch}/voice/derivatives
mkdir -p ${outdir}
#export SINGULARITYENV_TEMPLATEFLOW_HOME=$outdir/.templateflow

# index slurm array to grab subject
#subject=${subjs[${SLURM_ARRAY_TASK_ID}]}

echo Submitted subject: ${subject}

#set up singularity container
module add openmind/singularity
SING_IMG=/om2/user/rfbrito/containers/imaging/fmriprep_22.0.2.sif

# assign working directory
scratch=${myscratch}/fmriprep_work/${subject}
mkdir -p ${scratch}

#set fs license
export SINGULARITYENV_FS_LICENSE=/om2/user/dclb/imaging/fs_license_linux.txt

#set up
unset $PYTHONPATH
set -e

# default command
#cmd="singularity run -e -B /om2 -B ${scratch}:/workdir -B ${bids_dir}:/input:ro -B ${outdir}:/out ${SING_IMG} /input /out participant --participant_label ${subject} -w /workdir --cifti-output 91k --mem_mb 40000 --output-spaces MNI152NLin6Asym:res-2 anat --nprocs 16 --omp-nthreads 8"

cmd="singularity run -e -B /om2,${bids_dir}:/input:ro ${SING_IMG} /input ${outdir} participant --participant_label ${subject} -w ${scratch} --cifti-output 91k --mem_mb 40000 --nprocs 16 --omp-nthreads 8 --output-spaces MNI152NLin6Asym:res-2 T1w"

printf "Command:\n${cmd}\n"

echo DONE

#remove an IsRunning files from freesurfer
rm -f ${outdir}/freesurfer/${subject}/scripts/IsRunning*

# run it
eval $cmd

