#!/bin/sh
set -x #echo on
set -e #stop on errors

# HISF specification
SAMPLING=$1
SEED_RECOMP=$2
PATH_COST=$3
FEATS=$4
ALPHA=$5
BETA=$6
N_ITERS=$7
PRINT_OPT=2 # Print CSV data

# Working folders
ORIG_PATH=$8
PREV_SEG_PATH=$9
NEW_SEG_PATH=$10
GT_PATH=$11
GT_TYPE=$12

# Basic input validation
if [ -z $12 ]; then
  echo "Usage $0 <sampling> <seed-recomp> <path-cost> <feats> <alpha> <beta> <nIters> <orig-folder> <prev-seg-folder> <result-folder> <gt-folder> <gt-type(0=region,1=border)>"
  echo "See iftHISF_segmentation for parameter details."
  exit 1
fi

mkdir -p ${NEW_SEG_PATH}

for nSp in 50 100 150 200 250 300 350 400 450 500 ; do
  echo "time,nSp,br,ubr,bp,ubp,bf,ubf,asa,ue,comp,top,dice" > ${NEW_SEG_PATH}/${nSp}.csv
done

filenames=`find ${ORIG_PATH} -type f -printf "%f\n" | sort`
for file in $filenames ; do
  #echo $file
  basename="${file%.*}"
  sample_folder="${NEW_SEG_PATH}/seg_${basename}"
  mkdir -p ${sample_folder}
  sample_csv="${sample_folder}/results.csv"
  printf "time,nSp,br,ubr,bp,ubp,bf,ubf,asa,ue,comp,top,dice" > ${sample_csv}

  # Compute segmentations noting that effective number of superpixels is reduced
  for nSp in 25 50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000 1100 1200 1300 1400 1500 ; do
    printf "\n" >> ${sample_csv}

    # Actual segmentation
    iftRISF_segmentation --sampling ${SAMPLING} --seed-recomp ${SEED_RECOMP} --path-cost ${PATH_COST} --input-image ${ORIG_PATH}/${file} --output-image ${sample_folder}/${nSp}.pgm --superpixel-num ${nSp} --alpha ${ALPHA} --beta ${BETA} --iters-num ${N_ITERS} --superpixel-feat ${FEATS} --print-opt ${PRINT_OPT} --prev-seg ${PREV_SEG_PATH}/${basename}.pgm >> ${sample_csv}
    printf "," >> ${sample_csv}

    # Compute metrics for result in a global csv (for the sample)
    gt_prefix=$GT_PATH/${basename}
    iftComputeSuperpixelMetrics ${ORIG_PATH}/${file} ${gt_prefix} ${GT_TYPE} ${sample_folder}/${nSp}.pgm 0 ${PRINT_OPT} >> ${sample_csv}
  done

  # Occasionally increasing the base nSp parameter decreases the resulting number of superpixels
  iftSortCSV ${sample_csv} 1 ${sample_csv} 

  # Compute appropriate superpixel counts by interpolation
  for nSp in 50 100 150 200 250 300 350 400 450 500 ; do
    iftInterpolateSampleStats ${sample_csv} ${nSp} >> ${NEW_SEG_PATH}/${nSp}.csv 
    printf "\n" >> ${NEW_SEG_PATH}/${nSp}.csv
  done
done

# Result csv "header"
CSV_PATH=${NEW_SEG_PATH}/results.csv
echo "sampling,seedRecomp,pathCost,feats,baseNumSp,alpha,beta,nIters,time,stddev,nSp,stddev,br,stddev,ubr,stddev,bp,stddev,ubp,stddev,bf,stddev,ubf,stddev,asa,stddev,ue,stddev,comp,stddev,top,stddev,dice,stddev" > ${CSV_PATH}

# Aggregate results
for nSp in 50 100 150 200 250 300 350 400 450 500 ; do
  printf "%s,%s,%s,%s,%s,%s,%s,%s," ${SAMPLING} ${SEED_RECOMP} ${PATH_COST} ${FEATS} ${nSp} ${ALPHA} ${BETA} ${N_ITERS} >> ${CSV_PATH}
  iftAggregateCSVStats ${NEW_SEG_PATH}/${nSp}.csv 2 >> ${CSV_PATH}
  printf "\n" >> ${CSV_PATH}
done

