#!/bin/bash
#SBATCH --job-name=jupyter_notebook
#SBATCH --output=logs/notebook%j.out
#SBATCH --error=logs/notebook%j.err
#SBATCH --mem=100GB
#SBATCH -t 20:00:00
#SBATCH -p gablab
#SBATCH -c 14
#SBATCH -n 1
#SBATCH -x node[054-060,100-115]

source activate /om2/user/dclb/.miniconda/envs/nilearn_older_v

unset XDG_RUNTIME_DIR && jupyter notebook --no-browser --ip=0.0.0.0 --port=9400
