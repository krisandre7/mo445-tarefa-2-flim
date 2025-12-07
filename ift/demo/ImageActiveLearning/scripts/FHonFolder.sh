#!/bin/sh
set -x #echo on
set -e #stop on errors

# HISF specification
FEATS=$1
K_PARAM=$2
MIN_SIZE=$3
PRINT_OPT=2 # Print CSV data

# Working folders
ORIG_PATH=$4
NEW_SEG_PATH=$5
GT_PATH=$6
GT_TYPE=$7

# Basic input validation
if [ -z $6 ]; then
  echo "Usage $0 <feats (1=RGB,2=LAB_hist> <k (scale)> < <min_size> <orig-folder> <result-folder> [<gt-folder> <gt-type(0=region,1=border)>]"
  echo "See iftFH_segmentation for parameter details."
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
  iftFH_segmentation -i ${ORIG_PATH}/${file} -o ${NEW_SEG_PATH}/${basename}.pgm -k ${K_PARAM} -f ${FEATS} -s ${MIN_SIZE} -p ${PRINT_OPT} >> ${CSV_PATH}

  # Compute metrics if groundtruth is available
  if [ ! -z "$GT_PATH" ]; then
    printf "," >> ${CSV_PATH}
    gt_prefix=$GT_PATH/${basename}
    iftComputeSuperpixelMetrics ${ORIG_PATH}/${file} ${gt_prefix} ${GT_TYPE} ${NEW_SEG_PATH}/${basename}.pgm 0 ${PRINT_OPT} >> ${CSV_PATH}
  fi

  printf "\n" >> ${CSV_PATH}
done

