#!/bin/sh
set -x #echo on
set -e #stop on errors

if [ -z $NEWIFT_DIR ]; then
  echo "Error -- Requires the $$NEWIFT_DIR environment variable set to base IFT folder."
  exit 1
fi

if [ -z $5 ]; then
  echo "Usage $0 <orig-folder> <seg-folder> <seg-type(0=region,1=border)> <tgt_folder> <method_name>"
  exit 1
fi

# Input parameters
ORIG_PATH=$1 # To make parsing filenames easier
SEG_FOLDER=$2
SEG_TYPE=$3
TGT_FOLDER=$4
METHOD=$5

# -- Make sure any necessary program is compiled
IFT_LIB=${NEWIFT_DIR}/lib/libift.a
BIN_FOLDER=${NEWIFT_DIR}/bin
DEMO_FOLDER=${NEWIFT_DIR}/demo/ImageActiveLearning
OVERLAY_DEMO=iftOverlayBorders
FIX_DEMO=iftFixLabelMap
FORCE_DEMO=iftForceLabelMapConnectivity
if [ ! -f "${IFT_LIB}" ]; then
  make -C ${NEWIFT_DIR}
fi
if [ ! -f "${BIN_FOLDER}/${OVERLAY_DEMO}" ]; then
  make -C ${DEMO_FOLDER} ${OVERLAY_DEMO}
fi
if [ ! -f "${BIN_FOLDER}/${FIX_DEMO}" ]; then
  make -C ${DEMO_FOLDER} ${FIX_DEMO}
fi
if [ ! -f "${BIN_FOLDER}/${FORCE_DEMO}" ]; then
  make -C ${DEMO_FOLDER} ${FORCE_DEMO}
fi

filenames=`find ${ORIG_PATH} -type f -printf "%f\n" | sort`
for file in $filenames ; do
  basename="${file%.*}"
  orig=${ORIG_PATH}/$file
  seg=${SEG_FOLDER}/${basename}.pgm
  tgt=${TGT_FOLDER}/${basename}/${METHOD}.png

  mkdir -p ${TGT_FOLDER}/${basename}

  # Guarantee every _region_ segmentation is in the appropriate format [1,nLabels]
  # DOES NOT work with masks marked by 0 brightness pixels
  if [ $SEG_TYPE -eq 0 ]; then
    ${FIX_DEMO} ${seg}
  fi

  $OVERLAY_DEMO $orig $tgt $seg
done

