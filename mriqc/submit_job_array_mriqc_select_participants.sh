#!/bin/bash

#SBATCH -o log/%x-%A-%a.out
#SBATCH -e log/%x-%A-%a.err
#SBATCH --mail-user=dclb@mit.edu
#SBATCH --mail-type=ALL

#SBATCH -J mriqc_participants_submission_script

bids=../../.. 

# go to bids directory and grab all subjects we want to convert
pushd ${bids} > /dev/null
#subjs=($(ls sub-* -d))
subjs=(sub-MM177 sub-MM224 sub-MM256)
popd > /dev/null

# take the length of the array
# this will be useful for indexing later
len=$(expr ${#subjs[@]} - 1)


echo Spawning ${#subjs[@]} sub-job

# submit subject to mriqc for processing
sbatch --array=0-$len ss_mriqc_participants.sh $bids ${subjs[@]}
