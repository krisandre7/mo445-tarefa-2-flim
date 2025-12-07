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
START_NSP=$4
PATH_COST=$5
ALPHA=$6

if [ -z $6 ]; then
  echo "Usage $0 <orig-folder> <result-folder> <prev-seg-folder> <scale_0> <path-cost> <alpha>"
  exit 1
fi

# Make sure any necessary program is compiled
SCRIPT_FOLDER=${DEMO_FOLDER}/scripts

# Add fixed parameters
PARAMS="$ORIG_PATH $TGT_PATH $PREV_SEG_PATH"

# Add params for each scale
NSP=$START_NSP
while [ $NSP -ge 50 ]; do
  NSP=`expr $NSP \* 100 / 150`
  PARAMS="$PARAMS $NSP $PATH_COST $ALPHA"
done

${SCRIPT_PATH}/PRL_hierarchy_experiment.sh $PARAMS
