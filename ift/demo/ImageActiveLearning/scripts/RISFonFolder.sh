#!/bin/sh
set -x #echo on
set -e #stop on errors

if [ -z $NEWIFT_DIR ]; then
  echo "Error -- Requires the $$NEWIFT_DIR environment variable set to base IFT folder."
  exit 1
fi

# HISF specification
SAMPLING=$1
SEED_RECOMP=$2
PATH_COST=$3
FEATS=$4
NUM_SUPERPIXELS=$5
ALPHA=$6
BETA=$7
N_ITERS=$8
PRINT_OPT=2 # Print CSV data

# Working folders
ORIG_PATH=$9
PREV_SEG_PATH=$10
NEW_SEG_PATH=$11

# Basic input validation
if [ -z $11 ]; then
  echo "Usage $0 <sampling> <seed-recomp> <path-cost> <feats> <nSuperpixels> <alpha> <beta> <nIters> <orig-folder> <prev-seg-folder> <result-folder>"
  echo "See iftRISF_segmentation for parameter details."
  exit 1
fi

# Make sure any necessary program is compiled
IFT_LIB=${NEWIFT_DIR}/lib/libift.a
BIN_FOLDER=${NEWIFT_DIR}/bin
DEMO_FOLDER=${NEWIFT_DIR}/demo/ImageActiveLearning
SEG_DEMO=iftRISF_segmentation
FIX_DEMO=iftFixLabelMap
if [ ! -f "${IFT_LIB}" ]; then
  make -C ${NEWIFT_DIR}
fi
if [ ! -f "${BIN_FOLDER}/${SEG_DEMO}" ]; then
  make -C ${DEMO_FOLDER} ${SEG_DEMO}
fi
if [ ! -f "${BIN_FOLDER}/${FIX_DEMO}" ]; then
  make -C ${DEMO_FOLDER} ${FIX_DEMO}
fi

mkdir -p ${NEW_SEG_PATH}
# Result csv "header"
CSV_PATH=${NEW_SEG_PATH}/timing.csv
echo "file,time" > ${CSV_PATH}

filenames=`find ${ORIG_PATH} -type f -printf "%f\n" | sort`

for file in $filenames ; do
  #echo $file
  basename="${file%.*}"
  TGT="${NEW_SEG_PATH}/${basename}.pgm"
  # Segmentation + time 
  printf "%s," ${TGT} >> ${CSV_PATH}
  ${FIX_DEMO} ${PREV_SEG_PATH}/${basename}.pgm
  ${SEG_DEMO} -s ${SAMPLING} -r ${SEED_RECOMP} -c ${PATH_COST} -i ${ORIG_PATH}/${file} -o ${TGT} -n ${NUM_SUPERPIXELS} -a ${ALPHA} -b ${BETA} -t ${N_ITERS} -f ${FEATS} -p ${PRINT_OPT} -l ${PREV_SEG_PATH}/${basename}.pgm >> ${CSV_PATH}

  printf "\n" >> ${CSV_PATH}
done

