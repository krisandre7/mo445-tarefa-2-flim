#!/bin/sh
set -x #echo on
set -e #stop on errors

# Working folders
ORIG_PATH=$1
NEW_SEG_PATH=$2
PREV_SEG_PATH=$3
GT_PATH=$4
GT_TYPE=$5

# Basic input validation
if [ -z $3 ]; then
  echo "Usage $0 <orig-folder> <result-folder> <prev-seg-folder> [<gt-folder> <gt-type(0=region,1=border)>]"
  exit 1
fi

#make iftHISF_segmentation
#make iftComputeSuperpixelMetrics
#make iftAggregateCSVStats

mkdir -p ${NEW_SEG_PATH}
CSV_PATH=${NEW_SEG_PATH}/results.csv
if [ -z $GT_PATH ]; then
  echo "sampling,seedRecomp,pathCost,feats,baseNumSp,alpha,beta,nIters,time,stddev" > ${CSV_PATH}
else
  echo "sampling,seedRecomp,pathCost,feats,baseNumSp,alpha,beta,nIters,time,stddev,nSp,stddev,br,stddev,ubr,stddev,bp,stddev,ubp,stddev,bf,stddev,ubf,stddev,asa,stddev,ue,stddev,comp,stddev,top,stddev,dice,stddev" > ${CSV_PATH}
  if [ -z $GT_TYPE ]; then
    GT_TYPE=0
  fi
fi

for SAMPLING in 1 4 ; do
  for SEED_RECOMP in 2 ; do
    PATH_COST=${SEED_RECOMP}
    for FEATS in 1 ; do
      for NUM_SUPERPIXELS in 50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000 ; do
        for ALPHA in 3000.0 ; do
          for BETA in 12 ; do
            for N_ITERS in 10 ; do
              RES_FOLDER=${NEW_SEG_PATH}/${SAMPLING}_${SEED_RECOMP}_${PATH_COST}_${FEATS}_${NUM_SUPERPIXELS}_${ALPHA}_${BETA}_${N_ITERS}/
              echo ${RES_FOLDER}
              ./HISFonFolder.sh ${SAMPLING} ${SEED_RECOMP} ${PATH_COST} ${FEATS} ${NUM_SUPERPIXELS} ${ALPHA} ${BETA} ${N_ITERS} ${ORIG_PATH} ${PREV_SEG_PATH} ${RES_FOLDER} ${GT_PATH} ${GT_TYPE}
              printf "%s,%s,%s,%s,%s,%s,%s,%s," ${SAMPLING} ${SEED_RECOMP} ${PATH_COST} ${FEATS} ${NUM_SUPERPIXELS} ${ALPHA} ${BETA} ${N_ITERS} >> ${CSV_PATH}
              iftAggregateCSVStats ${RES_FOLDER}/sample_stats.csv 2 >> ${CSV_PATH}
              printf "\n" >> ${CSV_PATH} 
            done
          done
        done
      done
    done
  done
done

