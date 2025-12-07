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
NEW_SEG_PATH=$9
GT_PATH=$10
GT_TYPE=$11

# Basic input validation
if [ -z $11 ]; then
  echo "Usage $0 <sampling> <seed-recomp> <path-cost> <feats> <alpha> <beta> <nIters> <orig-folder> <result-folder> <gt-folder> <gt-type(0=region,1=border)>"
  echo "See iftHISF_segmentation for parameter details."
  exit 1
fi

mkdir -p ${NEW_SEG_PATH}

for nSp in 50 100 150 200 250 300 350 400 450 500 ; do
  echo "time,nSp,br,ue,comp,top,dice" > ${NEW_SEG_PATH}/${nSp}.csv
done

filenames=`find ${ORIG_PATH} -type f -printf "%f\n" | sort`
for file in $filenames ; do
  #echo $file
  basename="${file%.*}"
  sample_folder="${NEW_SEG_PATH}/seg_${basename}"
  mkdir -p ${sample_folder}
  sample_csv="${sample_folder}/results.csv"
  printf "time,nSp,br,ue,comp,top,dice" > ${sample_csv}

  # Compute the full hierarchy
  baseSp=10000
  prevSp=
  for nSp in ${baseSp} 1000 950 900 850 800 750 700 650 600 550 500 450 400 350 300 250 200 150 100 50 ; do
    printf "\n" >> ${sample_csv}

    # Actual segmentation, after the first we use our own segmentation recursively
    if [ -z $prevSp ]; then
      iftHISF_segmentation ${SAMPLING} ${SEED_RECOMP} ${PATH_COST} ${ORIG_PATH}/${file} ${sample_folder}/${nSp}.pgm ${nSp} ${ALPHA} ${BETA} ${N_ITERS} ${FEATS} ${PRINT_OPT} >> ${sample_csv}
    else
      iftHISF_segmentation ${SAMPLING} ${SEED_RECOMP} ${PATH_COST} ${ORIG_PATH}/${file} ${sample_folder}/${nSp}.pgm ${nSp} ${ALPHA} ${BETA} ${N_ITERS} ${FEATS} ${PRINT_OPT} ${sample_folder}/${baseSp}.pgm >> ${sample_csv}
    fi
    printf "," >> ${sample_csv}

    # Compute metrics for result in a global csv (for the sample)
    gt_prefix=$GT_PATH/${basename}
    iftComputeSuperpixelMetrics ${ORIG_PATH}/${file} ${gt_prefix} ${GT_TYPE} ${sample_folder}/${nSp}.pgm 0 ${PRINT_OPT} >> ${sample_csv}

    prevSp=$nSp
  done
  iftSortCSV ${sample_csv} 1 ${sample_csv} 

  # Compute appropriate superpixel counts by interpolation
  for nSp in 50 100 150 200 250 300 350 400 450 500 ; do
    iftInterpolateSampleStats ${sample_csv} ${nSp} >> ${NEW_SEG_PATH}/${nSp}.csv 
    printf "\n" >> ${NEW_SEG_PATH}/${nSp}.csv
  done
done

# Result csv "header"
CSV_PATH=${NEW_SEG_PATH}/results.csv
echo "sampling,seedRecomp,pathCost,feats,baseNumSp,alpha,beta,nIters,time,stddev,nSp,stddev,br,stddev,ue,stddev,comp,stddev,top,stddev,dice,stddev" > ${CSV_PATH}

# Aggregate results
for nSp in 50 100 150 200 250 300 350 400 450 500 ; do
  printf "%s,%s,%s,%s,%s,%s,%s,%s," ${SAMPLING} ${SEED_RECOMP} ${PATH_COST} ${FEATS} ${nSp} ${ALPHA} ${BETA} ${N_ITERS} >> ${CSV_PATH}
  iftAggregateCSVStats ${NEW_SEG_PATH}/${nSp}.csv 2 >> ${CSV_PATH}
  printf "\n" >> ${CSV_PATH}
done

