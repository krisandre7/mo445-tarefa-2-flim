#!/bin/sh
set -x #echo on
set -e #stop on errors

# Working folders
SNIC_BIN=snic
ORIG_PATH=$1
NEW_SEG_PATH=$2

# Basic input validation
if [ -z $2 ]; then
  echo "Usage $0 <orig-folder> <result-folder>]"
  exit 1
fi

mkdir -p ${NEW_SEG_PATH}
filenames=`find ${ORIG_PATH} -type f -printf "%f\n" | sort`
for nSp in 50 100 150 200 250 300 350 400 450 500 ; do
  SP_PATH=${NEW_SEG_PATH}/${nSp}
  mkdir -p ${SP_PATH}
  for file in $filenames ; do
    basename="${file%.*}"
    target=${SP_PATH}/${basename}.pgm
    compactness=10.0
    ${SNIC_BIN} ${ORIG_PATH}/${file} ${nSp} ${compactness} ${target}
    iftFixLabelMap ${target}
  done
done
