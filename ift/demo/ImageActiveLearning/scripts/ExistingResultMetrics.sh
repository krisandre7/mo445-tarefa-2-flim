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
BASE_SEG_FOLDER=$2
SEG_TYPE=$3
GT_PATH=$4
GT_TYPE=$5
METRICS_TYPE=$6
PRINT_CSVHEADER_PARAM="-p 4"
PRINT_CSVDATA_PARAM="2"

# -- Build fixed parameters string
# Should be identical to MetricsOnFolder.sh (TODO move to a shared call)
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
FOLDER_METRICS_SCRIPT=${DEMO_FOLDER}/scripts/MetricsOnFolder.sh
METRICS_DEMO=iftComputeSuperpixelMetrics
AGGREGATE_DEMO=iftAggregateCSVStats
SORT_DEMO=iftSortCSV
if [ ! -f "${IFT_LIB}" ]; then
  make -C ${NEWIFT_DIR}
fi
if [ ! -f "${BIN_FOLDER}/${METRICS_DEMO}" ]; then
  make -C ${DEMO_FOLDER} ${METRICS_DEMO}
fi
if [ ! -f "${BIN_FOLDER}/${AGGREGATE_DEMO}" ]; then
  make -C ${DEMO_FOLDER} ${AGGREGATE_DEMO}
fi
if [ ! -f "${BIN_FOLDER}/${SORT_DEMO}" ]; then
  make -C ${DEMO_FOLDER} ${SORT_DEMO}
fi

# Initialize CSV
if [ $METRICS_TYPE -eq 0 ]; then
  CSV_PATH=${BASE_SEG_FOLDER}/full_sp_metrics.csv
else
  CSV_PATH=${BASE_SEG_FOLDER}/full_ls_metrics.csv
fi
printf "folder," > ${CSV_PATH}
${METRICS_DEMO} -i dummy -g dummy ${PRINT_CSVHEADER_PARAM} ${FIXED_PARAMS} >> ${CSV_PATH}
printf "\n" >> ${CSV_PATH}

folders=`find ${BASE_SEG_FOLDER} -mindepth 1 -type d | sort`
for folder in ${folders} ; do
  ${FOLDER_METRICS_SCRIPT} ${ORIG_PATH} ${folder} ${SEG_TYPE} ${GT_PATH} ${GT_TYPE} ${METRICS_TYPE}

  # Standard names define in folder script
  if [ $METRICS_TYPE -eq 0 ]; then
    FOLDER_CSV_PATH=${folder}/sp_metrics.csv
  else
    FOLDER_CSV_PATH=${folder}/ls_metrics.csv
  fi

  printf "%s," $folder >> ${CSV_PATH}
  iftAggregateCSVStats ${FOLDER_CSV_PATH} ${PRINT_CSVDATA_PARAM} >> ${CSV_PATH} 
  printf "\n" >> ${CSV_PATH}
done

# By default column 1 is average number of regions/superpixels
${SORT_DEMO} ${CSV_PATH} 1 ${CSV_PATH}

