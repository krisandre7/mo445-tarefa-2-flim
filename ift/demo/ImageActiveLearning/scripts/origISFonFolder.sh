#!/bin/sh
# HISF specification
SAMPLING=$1
PATH_COST=$2
NUM_SUPERPIXELS=$3
ALPHA=$4
BETA=$5
NITERS=$6

# Working folders
ORIG_PATH=$7
NEW_SEG_PATH=$8

# Convert to ISF demo names
if [ "$SAMPLING" = 1 ]
then
  SAMPLING=GRID
else
  SAMPLING=MIX
fi

if [ "$PATH_COST" = 1 ]
then
  PATH_COST=ROOT
else
  PATH_COST=MEAN
fi

# Basic input validation
if [ -z $8 ]; then
  echo "Usage $0 <sampling> <path-cost> <nSuperpixels> <alpha> <beta> <niters> <orig-folder> <result-folder>"
  echo "See iftHISF_segmentation for parameter details."
  exit 1
fi

mkdir -p ${NEW_SEG_PATH}
filenames=`find ${ORIG_PATH} -type f -printf "%f\n" | sort`
for file in $filenames ; do
  echo $file
  basename="${file%.*}"
  iftISF_${SAMPLING}_${PATH_COST} ${ORIG_PATH}/${file} ${NUM_SUPERPIXELS} ${ALPHA} ${BETA} ${NITERS} 0 ${NEW_SEG_PATH}/${basename}
done

