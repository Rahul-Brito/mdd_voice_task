#!/bin/bash
#SBATCH --job-name=effectsize
#SBATCH --output=log/effectsize_%j.out
#SBATCH --error=log/effectsize_%j.err
#SBATCH --mem=32Gb
#SBATCH -N 1                 # one node
#SBATCH -n 4                 # two CPU (hyperthreaded) cores
#SBATCH --time=12:00:00
#SBATCH --partition=gablab
#SBATCH --mail-user=rfbrito@g.harvard.edu
#SBATCH --mail-type=ALL

module add openmind/miniconda

source activate /om2/user/rfbrito/miniconda/envs/imaging

python sub_level_effects_size_calculate.py
