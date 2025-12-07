#!/bin/sh
set -x #echo on
set -e #stop on errors

if [ -z $NEWIFT_DIR ]; then
  echo "Error -- Requires the $$NEWIFT_DIR environment variable set to base IFT folder."
  exit 1
fi

if [ -z $6 ]; then
  echo "Usage $0 <orig-folder> <seg-folder> <seg-type(0=region,1=border)> <gt-folder> <gt-type(0=region,1=border)> <metrics(0=superpixels,1=large-segment)"
  exit 1
fi

# Input parameters
ORIG_PATH=$1 # To make parsing filenames easier
SEG_FOLDER=$2
SEG_TYPE=$3
GT_FOLDER=$4
GT_TYPE=$5
METRICS_TYPE=$6
PRINT_CSVHEADER_PARAM="-p 3"
PRINT_CSVDATA_PARAM="-p 2"

# -- Build fixed parameters string
if [ $METRICS_TYPE -eq 0 ]; then
  FIXED_PARAMS="${FIXED_PARAMS} -sp"
else
  FIXED_PARAMS="${FIXED_PARAMS} -ls"
fi

if [ $SEG_TYPE -eq 1 ]; then
  FIXED_PARAMS="${FIXED_PARAMS} -ib"
fi

if [ $GT_TYPE -eq 1 ]; then
  FIXED_PARAMS="${FIXED_PARAMS} -gb"
fi

# -- Make sure any necessary program is compiled
IFT_LIB=${NEWIFT_DIR}/lib/libift.a
BIN_FOLDER=${NEWIFT_DIR}/bin
DEMO_FOLDER=${NEWIFT_DIR}/demo/ImageActiveLearning
METRICS_DEMO=iftComputeSuperpixelMetrics
FIX_DEMO=iftFixLabelMap
if [ ! -f "${IFT_LIB}" ]; then
  make -C ${NEWIFT_DIR}
fi
if [ ! -f "${BIN_FOLDER}/${METRICS_DEMO}" ]; then
  make -C ${DEMO_FOLDER} ${METRICS_DEMO}
fi
if [ ! -f "${BIN_FOLDER}/${FIX_DEMO}" ]; then
  make -C ${DEMO_FOLDER} ${FIX_DEMO}
fi

# Initialize CSV
if [ $METRICS_TYPE -eq 0 ]; then
  CSV_PATH=${SEG_FOLDER}/sp_metrics.csv
else
  CSV_PATH=${SEG_FOLDER}/ls_metrics.csv
fi
printf "file," > ${CSV_PATH}
${METRICS_DEMO} -i dummy -g dummy ${PRINT_CSVHEADER_PARAM} ${FIXED_PARAMS} >> ${CSV_PATH}
printf "\n" >> ${CSV_PATH}

filenames=`find ${ORIG_PATH} -type f -printf "%f\n" | sort`
for file in $filenames ; do
  basename="${file%.*}"
  seg=${SEG_FOLDER}/${basename}.pgm
  #gt_prefix=${GT_FOLDER}/${basename}

  # Temporary change for MSRC dataset
  gt_prefix=${GT_FOLDER}/${basename}

  # Guarantee every _region_ segmentation is in the appropriate format [1,nLabels]
  # DOES NOT work with masks marked by 0 brightness pixels
  if [ $SEG_TYPE -eq 0 ]; then
    ${FIX_DEMO} ${seg}
  fi

  printf "%s," $file >> ${CSV_PATH}
  ${METRICS_DEMO} -i ${seg} -g ${gt_prefix} ${PRINT_CSVDATA_PARAM} ${FIXED_PARAMS} >> ${CSV_PATH}
  printf "\n" >> ${CSV_PATH}
done

