#!/bin/sh
set -x #echo on

ORIG_PATH=$1
BASE_SEG_FOLDER=$2

if [ -z $2 ]; then
  echo "Usage $0 <orig-folder> <base-seg-folder>"
  exit 1
fi

folders=`find ${BASE_SEG_FOLDER} -mindepth 1 -type d | sort`
filenames=`find ${ORIG_PATH} -type f -printf "%f\n" | sort`

for folder in ${folders} ; do
  for file in ${filenames} ; do
    basename="${file%.*}"
    result=${folder}/${basename}.pgm
    iftPrintP5Header ${ORIG_PATH}/${file} > ${result}
    cat ${folder}/${basename}.dat >> ${result}
    iftFixLabelMap ${result}
  done
done

