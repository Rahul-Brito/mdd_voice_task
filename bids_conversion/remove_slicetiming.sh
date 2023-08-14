#!/bin/bash
#SBATCH --job-name=remove_slicetiming
#SBATCH --output=log/remove_slicetiming_%j.out
#SBATCH --error=log/remove_slicetiming_%j.err
#SBATCH --mem=32Gb
#SBATCH -N 1                 # one node
#SBATCH -n 2                 # two CPU (hyperthreaded) cores
#SBATCH --time=1:00:00
#SBATCH --partition=gablab
#SBATCH --mail-user=rfbrito@g.harvard.edu
#SBATCH --mail-type=ALL

module add openmind/miniconda

source activate /om2/user/rfbrito/miniconda/envs/voice_depression

python remove_slicetiming.py /om2/scratch/Wed/rfbrito/bids

