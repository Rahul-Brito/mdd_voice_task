#!/bin/bash

#SBATCH -o log/%x-%A-%a.out
#SBATCH -e log/%x-%A-%a.err
#SBATCH --mail-user=rfbrito@mit.edu
#SBATCH --mail-type=ALL
#SBATCH -J fmriprep_submission_script
#SBATCH --array=0-72%20

base=/om2/scratch/Wed/rfbrito/bids/
#session=baseline
#base=/om2/scratch/Wed/rfbrito/toy/

# go to dicom directory and grab all subjects we want to convert
pushd ${base} > /dev/null
subjs=($(ls sub-voice* -d))
popd > /dev/null

#subjs=sub-voice999

# take the length of the array
# this will be useful for indexing later
#len=$(expr ${#subjs[@]} - 1)


echo Spawning ${#subjs[@]} sub-job

# submit subject to fmriprep  processing
sbatch ss_fmriprep.sh $base ${subjs[$SLURM_ARRAY_TASK_ID]}
