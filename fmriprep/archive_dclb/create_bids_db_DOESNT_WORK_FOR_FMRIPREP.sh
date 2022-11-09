#!/bin/bash
#SBATCH -p use-everything
#SBATCH --time=01:00:00
#SBATCH --mem=5GB
#SBATCH -J create_bids_db
#SBATCH -o log/%x-%A-%a.out
#SBATCH -e log/%x-%A-%a.err
#SBATCH --mail-user=dclb@mit.edu
#SBATCH --mail-type=ALL


myscratch=/om2/scratch/Sat/$(whoami)
bids_dir=/om2/scratch/Sat/dclb/mct
outdir=${myscratch}/mct/derivatives

module add openmind/singularity/3.6.3
SING_IMG=/om2/user/dclb/containers/imaging/fmriprep_20.2.6.sif

unset $PYTHONPATH
set -e

mkdir -p ${bids_dir}/derivatives/bids_db

cmd="singularity exec -e -B ${bids_dir}:/input -B ${outdir}:/out $SING_IMG pybids layout /input /input/derivatives/bids_db --no-validate"

echo $cmd

# run it
eval $cmd



