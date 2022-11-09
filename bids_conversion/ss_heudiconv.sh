#!/bin/bash
#SBATCH -p gablab
#SBATCH --time=12:00:00
#SBATCH --mem=100GB
#SBATCH -J heudiconv
#SBATCH -o log/%x-%A-%a.out
#SBATCH -e log/%x-%A-%a.err
#SBATCH --mail-user=rfbrito@mit.edu
#SBATCH --mail-type=ALL

# grab these from submission script
base=$1
ses=$2
subject=$3


# set output for conversion
outdir=/om2/scratch/Wed/rfbrito/bids


echo Submitited subject: ${subject}
echo Submitted session: ${ses}

module add openmind/singularity
SING_IMG=/om2/user/rfbrito/containers/imaging/heudiconv_latest.sif

# default command
cmd="singularity run -B /om2/scratch/Wed/rfbrito/dicom -B /om2/user/rfbrito/voice_depression/bids_conversion -B ${outdir}:/out ${SING_IMG} -d /om2/scratch/Wed/rfbrito/dicom/{subject}/session00${ses}*/dicom/Trio*/*.dcm -o /out -f /om2/user/rfbrito/voice_depression/bids_conversion/heuristic.py -c dcm2niix -s ${subject} -ss ${ses} -b --minmeta -g accession_number"

# dry run command
#cmd="singularity run -B /mindhive -B ${outdir}:/out ${SING_IMG} -d /mindhive/xnat/dicom_storage/voice/dicom/{subject}/session00${ses}*/dicom/Trio*/*.dcm -o /out -f convertall -c none -s ${subject} -ss ${ses} --minmeta -g accession_number"

printf "Command:\n${cmd}\n"

# run it
eval $cmd

echo DONE
