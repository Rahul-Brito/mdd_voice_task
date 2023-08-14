#! /bin/bash -l
#SBATCH -p normal
#SBATCH --time=12:00:00
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=16
#SBATCH -J mriqc_participants
#SBATCH -o log/%x-%A-%a.out
#SBATCH -e log/%x-%A-%a.err
#SBATCH --mail-user=rfbrito@mit.edu
#SBATCH --mail-type=ALL

# grab these from submission script
bids_dir=$1
task=$2
args=($@)
subjs=(${args[@]:2}) #:1 needed because argument 1 is the input directory, which we need to exclude

myscratch=/om2/scratch/Fri/$(whoami)

# set output for conversion
outdir=${bids_dir}/derivatives/mriqc_per_task/${task}

# index slurm array to grab subject
subject=${subjs[${SLURM_ARRAY_TASK_ID}]}

echo Submitted subject: ${subject}

#set up singularity container
module add openmind/singularity
SING_IMG=/om2/scratch/Fri/rfbrito/mriqc-latest.sif

# assign working directory
scratch=${myscratch}/mriqc_work_${task}/${subject}
mkdir -p ${scratch}

#set up
unset $PYTHONPATH
set -e

# default command
cmd="singularity run -e -B /om2 -B ${scratch}:/workdir -B ${bids_dir}:/input:ro -B ${outdir}:/out ${SING_IMG} /input /out participant --task-id ${task} --participant_label ${subject} -w /workdir --nprocs 16 --omp-nthreads 8"

printf "Command:\n${cmd}\n"


# run it
eval $cmd
