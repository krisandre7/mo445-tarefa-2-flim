#!/bin/sh
set -x #echo on
set -e #stop on errors

# ISF specification
SAMPLING=$1
PATH_COST=$2
TYPE=$3
NUM_SUPERPIXELS=$4
ALPHA=$5

# Working folders
ORIG_PATH=$6
NEW_SEG_PATH=$7

# Basic input validation
if [ -z $7 ]; then
  echo "Usage $0 <sampling[0-grid,1-mixed]> <path-cost[0-mean,1-root]> <type[0-2Dgray,1-2Dcolor,2-3D]> <nSuperpixels> <alpha> <orig-folder> <result-folder>"
  exit 1
fi

if [ $TYPE -eq 2 ]; then
  GFT_ISF=${GFT_DIR}/bin/supervoxels
  RES_EXT=.scn
  N_OBJECTS=3 # Hardcoded for brain, may need to change later
else
  GFT_ISF=${GFT_DIR}/bin/superpixels
  RES_EXT=.pgm
  GFT_TYPE=$TYPE
fi

mkdir -p ${NEW_SEG_PATH}

# CSV path + header
CSV_PATH=${NEW_SEG_PATH}/timing.csv
echo "file,time" > ${CSV_PATH}

filenames=`find ${ORIG_PATH} -type f -printf "%f\n" | sort`
for file in $filenames ; do
  basename="${file%.*}"
  res=${NEW_SEG_PATH}/${basename}${RES_EXT}
  res_simple=${NEW_SEG_PATH}/${basename}
  
  # Segmentation + time
  printf "%s," ${res} >> ${CSV_PATH}
  ${GFT_ISF} ${GFT_TYPE} ${ORIG_PATH}/${file} ${NUM_SUPERPIXELS} ${ALPHA} ${SAMPLING} ${PATH_COST} ${res_simple} >> ${CSV_PATH}
  printf "\n" >> ${CSV_PATH}
done

