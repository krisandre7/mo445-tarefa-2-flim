#!/bin/sh
set -x #echo on
set -e #stop on errors

if [ -z $GFT_DIR ]; then
  echo "Error -- Requires the $$GFT_DIR environment variable set to isf_demo folder."
  exit 1
fi
if [ -z $NEWIFT_DIR ]; then
  echo "Error -- Requires the $$NEWIFT_DIR environment variable set to base IFT folder."
  exit 1
fi

# Working folders
ORIG_PATH=$1
NEW_SEG_PATH=$2
GT_PATH=$3
GT_TYPE=$4

# Basic input validation
if [ -z $2 ]; then
  echo "Usage $0 <orig-folder> <result-folder>"
  exit 1
fi

# Standard paths
DEMO_FOLDER=${NEWIFT_DIR}/demo/ImageActiveLearning
SCRIPT_PATH=${DEMO_FOLDER}/scripts

mkdir -p ${NEW_SEG_PATH}

# 0 = grid 1 = mixed
SAMPLING=0
# 0 = mean 1 = root
for PATH_COST in 0 ; do
  for ALPHA in 0.2 ; do
    PARAM_FOLDER=${NEW_SEG_PATH}/PATHCOST_${PATH_COST}_ALPHA_${ALPHA}/
    mkdir -p ${PARAM_FOLDER}
    #for NUM_SUPERPIXELS in 50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 1600 3200 6400; do
    for NUM_SUPERPIXELS in 6400; do
      RES_FOLDER=${PARAM_FOLDER}/seg_${NUM_SUPERPIXELS}/
      ${SCRIPT_PATH}/FastISFonFolder.sh ${SAMPLING} ${PATH_COST} 1 ${NUM_SUPERPIXELS} ${ALPHA} ${ORIG_PATH} ${RES_FOLDER}
    done
  done
done

