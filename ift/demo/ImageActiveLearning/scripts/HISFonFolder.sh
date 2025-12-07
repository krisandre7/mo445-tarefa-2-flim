#!/bin/sh
set -x #echo on
set -e #stop on errors

# HISF specification
SAMPLING=$1
SEED_RECOMP=$2
PATH_COST=$3
FEATS=$4
NUM_SUPERPIXELS=$5
ALPHA=$6
BETA=$7
N_ITERS=$8
PRINT_OPT=2 # Print CSV data

# Working folders
ORIG_PATH=$9
PREV_SEG_PATH=$10
NEW_SEG_PATH=$11
GT_PATH=$12
GT_TYPE=$13

# Basic input validation
if [ -z $11 ]; then
  echo "Usage $0 <sampling> <seed-recomp> <path-cost> <feats> <nSuperpixels> <alpha> <beta> <nIters> <orig-folder> <prev-seg-folder> <result-folder> [<gt-folder> <gt-type(0=region,1=border)>]"
  echo "See iftHISF_segmentation for parameter details."
  exit 1
fi

mkdir -p ${NEW_SEG_PATH}
# Result csv "header"
CSV_PATH=${NEW_SEG_PATH}/sample_stats.csv
if [ -z $GT_PATH ]; then
  echo "time" > ${CSV_PATH}
else
  echo "time,nSp,br,ubr,bp,ubp,bf,ubf,asa,ue,comp,top,dice" > ${CSV_PATH}
  if [ -z $GT_TYPE ]; then
    GT_TYPE=0
  fi
fi

filenames=`find ${ORIG_PATH} -type f -printf "%f\n" | sort`

for file in $filenames ; do
  #echo $file
  basename="${file%.*}"
  # Segmentation + time 
  iftHISF_segmentation ${SAMPLING} ${SEED_RECOMP} ${PATH_COST} ${ORIG_PATH}/${file} ${NEW_SEG_PATH}/${basename}.pgm ${NUM_SUPERPIXELS} ${ALPHA} ${BETA} ${N_ITERS} ${FEATS} ${PRINT_OPT} ${PREV_SEG_PATH}/${basename}.pgm >> ${CSV_PATH}

  # Compute metrics if groundtruth is available
  if [ ! -z "$GT_PATH" ]; then
    printf "," >> ${CSV_PATH}
    gt_prefix=$GT_PATH/${basename}
    iftComputeSuperpixelMetrics ${ORIG_PATH}/${file} ${gt_prefix} ${GT_TYPE} ${NEW_SEG_PATH}/${basename}.pgm 0 ${PRINT_OPT} >> ${CSV_PATH}
  fi

  printf "\n" >> ${CSV_PATH}
  #rm ${NEW_SEG_PATH}/${basename}.pgm
done

