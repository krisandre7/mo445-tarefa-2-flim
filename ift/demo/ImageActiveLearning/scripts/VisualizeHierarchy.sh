#!/bin/sh
set -x #echo on
set -e #stop on errors

if [ -z $NEWIFT_DIR ]; then
  echo "Error -- Requires the $$NEWIFT_DIR environment variable set to base IFT folder."
  exit 1
fi

# Standard interface
SCRIPT_PATH=${NEWIFT_DIR}/demo/ImageActiveLearning/scripts

QT_DEMO=${NEWIFT_DIR}/demo/Qt/InteractiveSuperpixels/InteractiveSuperpixels

if [ -z $4 ]; then
  echo "Usage $0 <orig-img> <hierarchy_path> <label_map_name> <base_nsp>"
  exit 1
fi

# Make sure any necessary program is compiler
IFT_LIB=${NEWIFT_DIR}/lib/libift.a
BIN_FOLDER=${NEWIFT_DIR}/bin
DEMO_FOLDER=${NEWIFT_DIR}/demo/ImageActiveLearning
SCRIPT_FOLDER=${DEMO_FOLDER}/scripts

ORIG_PATH=$1
HIERARCHY_PATH=$2
LABEL_MAP_NAME=$3
PREFIX=$HIERARCHY_PATH/seg_
POSTFIX=_2_0.5/$LABEL_MAP_NAME

# Add params for each scale
NSCALES=0
NSP=$4
while [ $NSP -ge 100 ]; do
  NSP=`expr $NSP \* 100 / 200`
  NSCALES=`expr $NSCALES + 1`
  PARAMS="$PARAMS ${PREFIX}${NSP}${POSTFIX}"
done

${QT_DEMO} ${ORIG_PATH} ${NSCALES} ${PARAMS} 

