#!/bin/sh
# Working folders
ORIG_PATH=$1
GT_PATH=$2
SEG_PATH=$3

# Basic input validation
if [ -z $3 ]; then
  echo "Usage $0 <orig-folder> <gt-folder> <result-folder>"
  exit 1
fi

foldernames=`find ${SEG_PATH}/* -maxdepth 0 | sort`
for folder in $foldernames ; do
  echo $folder
  iftComputeSuperpixelMetrics ${ORIG_PATH} ${GT_PATH} ${folder} result
done
