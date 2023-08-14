#! /bin/bash -l
#SBATCH -t 16:00:00
#SBATCH -n 1
#SBATCH -p normal
#
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=5G
#
#SBATCH -o log/%x-%A-%a.out
#SBATCH -e log/%x-%A-%a.err
#SBATCH --mail-user=rfbrito@mit.edu
#SBATCH --mail-type=ALL
# 
#SBATCH --job-name=mriqc_group


#set up input and output paths
task=nwr
bids_dir="/om2/scratch/Fri/rfbrito/bids"
out_dir="${bids_dir}/derivatives/mriqc_per_task/${task}"

#set up working directory
scratch=/om2/scratch/Fri/$(whoami)/mriqc_work_per_part/${task}
mkdir -p ${scratch}
#scratch=/om2/scratch/Thu/rfbrito/mriqc_group_work

#set up singularity container
module add openmind/singularity

SING_IMG=/om2/scratch/Fri/rfbrito/mriqc-latest.sif

#set up
unset $PYTHONPATH
set -e

# default command
#cmd="singularity run -e -B /om2 -B /om2/scratch/Thu/rfbrito/mriqc_group_work  -B /om2/scratch/Thu/rfbrito/bids /om2/user/rfbrito/containers/imaging/mriqc-032423.sif /om2/scratch/Thu/rfbrito/bids /om2/scratch/Thu/rfbrito/bids/derivatives/mriqc participant -w /om2/scratch/Thu/rfbrito/mriqc_group_work --nprocs 16 --omp-nthreads 8 --verbose"


#cmd="singularity run -e -B /om2 -B ${scratch}:/workdir -B ${bids_dir}:/input:ro -B ${out_dir}:/out ${SING_IMG} /input /out participant -w /workdir --nprocs 16 --omp-nthreads 8 --verbose"

cmd="singularity run -e -B /om2 -B ${scratch}:/workdir -B ${bids_dir}:/input:ro -B ${out_dir}:/out ${SING_IMG} /input /out group -w /workdir --nprocs 16 --omp-nthreads 8 --verbose"

printf "Command:\n${cmd}\n"

#run it
eval $cmd

