#!/bin/sh
set -x #echo on
set -e #stop on errors

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
NEW_SEG_PATH=$10
GT_PATH=$11
GT_TYPE=$12

# Basic input validation
if [ -z $10 ]; then
  echo "Usage $0 <sampling> <seed-recomp> <path-cost> <feats> <nSuperpixels> <alpha> <beta> <nIters> <orig-folder> <result-folder> [<gt-folder> <gt-type(0=region,1=border)>]"
  echo "See iftHISF_segmentation for parameter details."
  exit 1
fi

# Fixed for now, may add customization later
SP_METRICS="-sp -sg"

# Make sure any necessary program is compiled
IFT_LIB=${NEWIFT_DIR}/lib/libift.a
BIN_FOLDER=${NEWIFT_DIR}/bin
DEMO_FOLDER=${NEWIFT_DIR}/demo/ImageActiveLearning
SEG_DEMO=iftHISF_segmentation
FIX_DEMO=iftForceLabelMapConnectivity
METRICS_DEMO=iftComputeSuperpixelMetrics
if [ ! -f "${IFT_LIB}" ]; then
  make -C ${NEWIFT_DIR}
fi
if [ ! -f "${BIN_FOLDER}/${SEG_DEMO}" ]; then
  make -C ${DEMO_FOLDER} ${SEG_DEMO}
fi
if [ ! -f "${BIN_FOLDER}/${METRICS_DEMO}" ]; then
  make -C ${DEMO_FOLDER} ${METRICS_DEMO}
fi
if [ ! -f "${BIN_FOLDER}/${FIX_DEMO}" ]; then
  make -C ${DEMO_FOLDER} ${FIX_DEMO}
fi

mkdir -p ${NEW_SEG_PATH}
# Result csv "header"
CSV_PATH=${NEW_SEG_PATH}/sample_stats.csv
if [ -z $GT_PATH ]; then
  echo "file,time" > ${CSV_PATH}
else
<<<<<<< .mine
  printf "file,time," > ${CSV_PATH}
  ${METRICS_DEMO} -i dummy -g dummy -p 3 ${SP_METRICS} >> ${CSV_PATH}
  printf "\n" >> ${CSV_PATH}
||||||| .r4315
  printf "file,time," > ${CSV_PATH}
  ${METRICS_DEMO} -i dummy -g dummy-p 3 ${SP_METRICS} >> ${CSV_PATH}
  printf "\n" >> ${CSV_PATH}
=======
  echo "time,nSp,br,ubr,bp,ubp,bf,ubf,asa,ue,comp,top,dice" > ${CSV_PATH}
>>>>>>> .r4216
  if [ -z $GT_TYPE ]; then
    GT_TYPE=0
  fi
fi

filenames=`find ${ORIG_PATH} -type f -printf "%f\n" | sort`

for file in $filenames ; do
  #echo $file
  basename="${file%.*}"
  TGT="${NEW_SEG_PATH}/${basename}.pgm"
  # Segmentation + time 
  printf "%s," ${TGT} >> ${CSV_PATH}
  ${SEG_DEMO} ${SAMPLING} ${SEED_RECOMP} ${PATH_COST} ${ORIG_PATH}/${file} ${TGT} ${NUM_SUPERPIXELS} ${ALPHA} ${BETA} ${N_ITERS} ${FEATS} ${PRINT_OPT} >> ${CSV_PATH}

  # While we do not find the bug generating isolated pixels
  ${FIX_DEMO} ${TGT} ${TGT} 10 

  # Compute metrics if groundtruth is available
  if [ ! -z "$GT_PATH" ]; then
    printf "," >> ${CSV_PATH}
    gt_prefix=$GT_PATH/${basename}
    ${METRICS_DEMO} -g ${gt_prefix} -i ${TGT} -p ${PRINT_OPT} -gb ${GT_TYPE} ${SP_METRICS} >> ${CSV_PATH}
  fi

  printf "\n" >> ${CSV_PATH}
done

