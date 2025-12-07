#!/bin/sh
#set -x #echo on
set -e #stop on errors

TGT_FOLDER=$1
INDEX_START=$2
SEC_INDEX=$3

if [ -z $1 ]; then
  echo "Usage $0 <target_folder> [Index_start] [Secondary_index]"
  exit 1
fi

if [ -z $2 ]; then
  INDEX_START=1
fi

if [ -z $3 ]; then
  SEC_INDEX=1
fi

filenames=`find ${TGT_FOLDER} -type f -printf "%f\n" | sort`
MAIN_INDEX=$INDEX_START

for file in $filenames ; do
  TGT=`printf %s/%06d_%06d.%s ${TGT_FOLDER} ${SEC_INDEX} ${MAIN_INDEX} ${file##*.}`
  #echo $TGT
  mv ${TGT_FOLDER}/$file $TGT
  MAIN_INDEX=`expr $MAIN_INDEX + 1`
done
