#!/bin/sh
set -x #echo on
set -e #stop on errors

if [ -z $4 ]; then
  echo "Usage $0 <l0-sample-stats.csv> <target-l0-time.csv> <l1-result.csv> <l1-time-col>"
  exit 1
fi

L0_SAMPLE_STATS=$1
L0_TIME=$2
L1_RESULT=$3
L1_TIME_COL=$4

echo time,stddev > ${L0_TIME}
iftAggregateCSVStats ${L0_SAMPLE_STATS} 2 >> ${L0_TIME}
iftAddPreviousLevelsTimeToCSV ${L1_RESULT} ${L1_TIME_COL} ${L1_RESULT} ${L0_TIME} 0

