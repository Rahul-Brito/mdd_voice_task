#! /bin/bash

#SBATCH -o log/%x-%A-%a.out
#SBATCH -t 05:00:00
#SBATCH -e log/%x-%A-%a.err
#SBATCH --mail-user=rfbrito@mit.edu
#SBATCH --mail-type=ALL

#SBATCH -J mriqc_participants_submission_script

task=vowel
bids=/om2/scratch/Fri/rfbrito/bids

# go to bids directory and grab all subjects we want to convert
pushd ${bids} > /dev/null
subjs=($(ls sub-* -d))
#subjs=($(ls sub-* -d| tr -d sub-)) # need to remove sub- for mriqc
popd > /dev/null

# take the length of the array
# this will be useful for indexing later
len=$(expr ${#subjs[@]} - 1)


echo Spawning ${#subjs[@]} sub-job

# submit subject to mriqc for processing
sbatch --array=0-$len ss_mriqc_participants.sh $bids $task ${subjs[@]}
