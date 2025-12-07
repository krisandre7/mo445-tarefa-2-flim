#!/bin/sh
set -x #echo on
set -e #stop on errors

#
BASE_IMG=$1
TGT_PREFIX=$2

if [ -z $3 ]; then
  echo "Usage $0 <base_img> <output_prefix> <finest scale> [2nd finest scale...]"
  exit 1
fi

shift 2
SCALES=$@

SCALE_NUM=1
for scale in $SCALES ; do
  TGT="${TGT_PREFIX}_${SCALE_NUM}.pgm"
  
  iftRISF_segmentation -i $BASE_IMG -o $TGT -n $scale $PREV_SEG_PARAM

  PREV_SEG_PARAM="-l $TGT"
  SCALE_NUM=`expr $SCALE_NUM + 1`
done

