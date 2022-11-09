#!/bin/bash
#SBATCH --job-name=add_intendedfor
#SBATCH --output=log/add_intendedfor_%j.out
#SBATCH --error=log/add_intendedfor_%j.err
#SBATCH --mem=32Gb
#SBATCH -N 1                 # one node
#SBATCH -n 2                 # two CPU (hyperthreaded) cores
#SBATCH --time=1:00:00
#SBATCH --partition=gablab
#SBATCH --mail-user=rfbrito@g.harvard.edu
#SBATCH --mail-type=ALL

#module purge

module add openmind/miniconda

source activate /om2/user/rfbrito/miniconda/envs/voice_depression

#conda activate /om2/user/rfbrito/miniconda/envs/voice_depression

python3  add_intendedfor.py /om2/scratch/Wed/rfbrito/bids

