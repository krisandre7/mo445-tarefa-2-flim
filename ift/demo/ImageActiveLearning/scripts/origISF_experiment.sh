#!/bin/sh
# Working folders
ORIG_PATH=$1
NEW_SEG_PATH=$2

# Basic input validation
if [ -z $2 ]; then
  echo "Usage $0 <orig-folder> <result-folder>"
  exit 1
fi

mkdir -p ${NEW_SEG_PATH}
for SAMPLING in 1 ; do
  for PATH_COST in 1 ; do
    for NUM_SUPERPIXELS in 100 150 200 250 300 350 400 450 500 ; do
      for ALPHA in 0.5 ; do
        for BETA in 12 ; do
          RES_FOLDER=${NEW_SEG_PATH}/${SAMPLING}_${PATH_COST}_${NUM_SUPERPIXELS}_${ALPHA}_${BETA}/
          echo ${RES_FOLDER}
          ./origISFonFolder.sh ${SAMPLING} ${PATH_COST} ${NUM_SUPERPIXELS} ${ALPHA} ${BETA} 10 ${ORIG_PATH} ${RES_FOLDER}
        done
      done
    done
  done
done

