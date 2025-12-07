#!/bin/sh
set -x #echo on
set -e #stop on errors

if [ -z $NEWIFT_DIR ]; then
  echo "Error -- Requires the $$NEWIFT_DIR environment variable set to base IFT folder."
  exit 1
fi

if [ -z $3 ]; then
  echo "Usage $0 <orig-folder> <seg-folder> <vis-folder>"
  exit 1
fi

# Input parameters
ORIG_PATH=$1
SEG_FOLDER=$2
VIS_FOLDER=$3

# -- Make sure any necessary program is compiled
IFT_LIB=${NEWIFT_DIR}/lib/libift.a
BIN_FOLDER=${NEWIFT_DIR}/bin
DEMO_FOLDER=${NEWIFT_DIR}/demo/ImageActiveLearning
CONN_DEMO=iftShowConnectivityProblems
FIX_DEMO=iftFixLabelMap
if [ ! -f "${IFT_LIB}" ]; then
  make -C ${NEWIFT_DIR}
fi
if [ ! -f "${BIN_FOLDER}/${CONN_DEMO}" ]; then
  make -C ${DEMO_FOLDER} ${CONN_DEMO}
fi
if [ ! -f "${BIN_FOLDER}/${FIX_DEMO}" ]; then
  make -C ${DEMO_FOLDER} ${FIX_DEMO}
fi

filenames=`find ${ORIG_PATH} -type f -printf "%f\n" | sort`
for file in $filenames ; do
  basename="${file%.*}"
  seg=${SEG_FOLDER}/${basename}.pgm

  # Guarantee every _region_ segmentation is in the appropriate format [1,nLabels]
  # DOES NOT work with masks marked by 0 brightness pixels
  ${FIX_DEMO} ${seg}

  ${CONN_DEMO} $ORIG_PATH/$file $seg $VIS_FOLDER/$basename 
done

