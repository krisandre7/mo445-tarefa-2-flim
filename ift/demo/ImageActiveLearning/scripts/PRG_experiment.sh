#!/bin/sh
set -x #echo on
set -e #stop on errors

if [ -z $NEWIFT_DIR ]; then
  echo "Error -- Requires the $$NEWIFT_DIR environment variable set to base IFT folder."
  exit 1
fi

# Working folders
ORIG_PATH=$1
BASE_SEG_PATH=$2
BASE_NSP=$3
NEW_SEG_PATH=$4

# Basic input validation
if [ -z $3 ]; then
  echo "Usage $0 <orig-folder> <base-seg> <base-nSp> <result-folder>"
  exit 1
fi


# Make sure any necessary program is compiled
IFT_LIB=${NEWIFT_DIR}/lib/libift.a
BIN_FOLDER=${NEWIFT_DIR}/bin
DEMO_FOLDER=${NEWIFT_DIR}/demo/ImageActiveLearning
SCRIPT_PATH=${DEMO_FOLDER}/scripts
PRG_DEMO=iftPRG_segmentation
if [ ! -f "${IFT_LIB}" ]; then
  make -C ${NEWIFT_DIR}
fi
if [ ! -f "${BIN_FOLDER}/${PRG_DEMO}" ]; then
  make -C ${DEMO_FOLDER} ${PRG_DEMO}
fi

mkdir -p ${NEW_SEG_PATH}
# Copy first level for convenience
cp -r $BASE_SEG_PATH $NEW_SEG_PATH/seg_orig
DELTA_SP=1
MIN_SP=2

filenames=`find ${ORIG_PATH} -type f -printf "%f\n" | sort`

CUR_SP=$BASE_NSP
while [ $CUR_SP -ge $MIN_SP ] ; do
  RES_FOLDER="${NEW_SEG_PATH}/seg_${CUR_SP}"
  mkdir -p $RES_FOLDER
  
  # Prepare per sample CSV
  CSV_PATH="${RES_FOLDER}/timing.csv"
  echo "file,time" > ${CSV_PATH}

  for file in $filenames ; do
    ORIG_IMG=${ORIG_PATH}/${file}
    basename="${file%.*}"
    TGT="${RES_FOLDER}/${basename}.pgm"
    PRINT_OPT=2 # Print csv data

    COMMON_PARS="-i $ORIG_IMG -o $TGT -n $CUR_SP -p ${PRINT_OPT}"

    PREV_SEG="${BASE_SEG_PATH}/${basename}.pgm"
    ${PRG_DEMO} ${COMMON_PARS} -l $PREV_SEG >> ${CSV_PATH}

    printf "\n" >> ${CSV_PATH}
  done

  CUR_SP=`expr $CUR_SP - $DELTA_SP`
  BASE_SEG_PATH=${RES_FOLDER}
done

