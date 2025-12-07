#!/bin/sh
set -x #echo on
set -e #stop on errors

if [ -z $NEWIFT_DIR ]; then
  echo "Error -- Requires the $$NEWIFT_DIR environment variable set to base IFT folder."
  exit 1
fi

# Working folders
ORIG_PATH=$1
NEW_SEG_PATH=$2
PREV_SEG_PATH=$3

# Basic input validation
if [ -z $3 ]; then
  echo "Usage $0 <orig-folder> <result-folder> <prev-seg-folder>"
  exit 1
fi

# Make sure any necessary program is compiler
IFT_LIB=${NEWIFT_DIR}/lib/libift.a
DEMO_FOLDER=${NEWIFT_DIR}/demo/ImageActiveLearning
SCRIPT_PATH=${DEMO_FOLDER}/scripts

mkdir -p ${NEW_SEG_PATH}
for SAMPLING in 4 ; do
  for SEED_RECOMP in 2 ; do
    PATH_COST=${SEED_RECOMP}
    for FEATS in 1 ; do
      for ALPHA in 0.5 ; do
        for BETA in 12 ; do
          for N_ITERS in 10 ; do
            PARAM_FOLDER=${NEW_SEG_PATH}/SAMPLING_${SAMPLING}_PATHCOST_${PATH_COST}_ALPHA_${ALPHA}/
            for NUM_SUPERPIXELS in 50 100 200 300 400 500 600 700 800 900 1000 ; do
              RES_FOLDER=${PARAM_FOLDER}/seg_${NUM_SUPERPIXELS}/
              ${SCRIPT_PATH}/RISFonFolder.sh ${SAMPLING} ${SEED_RECOMP} ${PATH_COST} ${FEATS} ${NUM_SUPERPIXELS} ${ALPHA} ${BETA} ${N_ITERS} ${ORIG_PATH} ${PREV_SEG_PATH} ${RES_FOLDER}
            done
          done
        done
      done
    done
  done
done

