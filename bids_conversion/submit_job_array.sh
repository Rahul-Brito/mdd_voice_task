#!/bin/bash

#SBATCH -o log/%x-%A-%a.out
#SBATCH -e log/%x-%A-%a.err
#SBATCH --mail-user=rfbrito@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0-72%10
#SBATCH -J heudiconv_submission_script

base=/om2/scratch/Wed/rfbrito/dicom/
#base=/mindhive/xnat/dicom_storage/voice/dicom/

# go to dicom directory and grab all subjects we want to convert
pushd ${base} > /dev/null
subjs=($(ls voice* -d))
popd > /dev/null

#subjs=(voice875) #voice994)

# take the length of the array
# this will be useful for indexing later
#len=$(expr ${#subjs[@]} - 1)


ses=(1 2)

#echo One subject is  ${subjs[$SLURM_ARRAY_TASK_ID]}
#echo The slurm Id is $SLURM_ARRAY_TASK_ID
#echo The sessions are ${ses[$SLURM_ARRAY_TASK_ID]}

# submit subject to heudiconv processing
#echo ${subjs[$SLURM_ARRAY_TASK_ID]}
for s in ${ses[@]}; do
	sbatch ss_heudiconv.sh $base $s ${subjs[$SLURM_ARRAY_TASK_ID]}
#	sbatch ss_heudiconv.sh $base $s ${subjs[$SLURM_ARRAY_TASK_ID]}
done
