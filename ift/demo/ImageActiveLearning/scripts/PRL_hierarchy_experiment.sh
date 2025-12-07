#!/bin/sh
set -x #echo on
set -e #stop on errors

if [ -z $NEWIFT_DIR ]; then
  echo "Error -- Requires the $$NEWIFT_DIR environment variable set to base IFT folder."
  exit 1
fi

# Standard interface
SCRIPT_PATH=${NEWIFT_DIR}/demo/ImageActiveLearning/scripts
ORIG_PATH=$1
TGT_PATH=$2
PREV_SEG_PATH=$3

if [ -z $6 ]; then
  echo "Usage $0 <orig-folder> <result-folder> <prev-seg-folder> [<scale_n> <path-cost_n> <alpha_n>] n > 0 times"
  exit 1
fi

# Make sure any necessary program is compiler
IFT_LIB=${NEWIFT_DIR}/lib/libift.a
BIN_FOLDER=${NEWIFT_DIR}/bin
DEMO_FOLDER=${NEWIFT_DIR}/demo/ImageActiveLearning
SCRIPT_FOLDER=${DEMO_FOLDER}/scripts

mkdir -p ${TGT_PATH}

# From here we consume the options for each scale
shift 3
while [ $1 ]; do
  if [ -z $3 ]; then
    echo "Each scale needs <nSuperpixels> <path-cost(1=ROOT,2=MEAN)> <alpha>"
    exit 1
  fi
  
  # Segmentation parameters
  NUM_SUPERPIXELS=$1
  SAMPLING=1 # 1 = Grid 2 = Mixed 4 = Geodesic
  SEED_RECOMP=$2 # 1 = Root(medoid) 2 = Mean(centroid)
  PATH_COST=$2 # 1 = Root 2 = Mean
  ALPHA=$3
  BETA=12
  N_ITERS=10 # Was 2?
  FEATS=1 # 1 = LABmean
  RES_FOLDER=${TGT_PATH}/PATHCOST_${2}_ALPHA_${3}/seg_${1}/

  # Execute segmentation
  ${SCRIPT_FOLDER}/RISFonFolder.sh ${SAMPLING} ${SEED_RECOMP} ${PATH_COST} ${FEATS} ${NUM_SUPERPIXELS} ${ALPHA} ${BETA} ${N_ITERS} ${ORIG_PATH} ${PREV_SEG_PATH} ${RES_FOLDER}

  # Update indexes for next scale
  PREV_SEG_PATH=$RES_FOLDER
  shift 3
done

