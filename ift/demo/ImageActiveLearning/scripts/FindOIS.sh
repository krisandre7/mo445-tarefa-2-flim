#!/bin/sh
set -x #echo on
set -e #stop on errors

if [ -z $NEWIFT_DIR ]; then
  echo "Error -- Requires the $$NEWIFT_DIR environment variable set to base IFT folder."
  exit 1
fi

if [ -z $1 ]; then
  echo "Usage $0 <seg_folder>"
  exit 1
fi

BASE_SEG_FOLDER=$1

# -- Make sure any necessary program is compiled
IFT_LIB=${NEWIFT_DIR}/lib/libift.a
BIN_FOLDER=${NEWIFT_DIR}/bin
DEMO_FOLDER=${NEWIFT_DIR}/demo/ImageActiveLearning
CSVUPDATE_DEMO=iftUpdateBestSampleStatCSV
AGGREGATE_DEMO=iftAggregateCSVStats
METRICS_DEMO=iftComputeSuperpixelMetrics
if [ ! -f "${IFT_LIB}" ]; then
  make -C ${NEWIFT_DIR}
fi
if [ ! -f "${BIN_FOLDER}/${CSVUPDATE_DEMO}" ]; then
  make -C ${DEMO_FOLDER} ${CSVUPDATE_DEMO}
fi
if [ ! -f "${BIN_FOLDER}/${AGGREGATE_DEMO}" ]; then
  make -C ${DEMO_FOLDER} ${AGGREGATE_DEMO}
fi
if [ ! -f "${BIN_FOLDER}/${METRICS_DEMO}" ]; then
  make -C ${DEMO_FOLDER} ${METRICS_DEMO}
fi

FIRST=1
folders=`find ${BASE_SEG_FOLDER} -mindepth 1 -type d | sort`
for folder in ${folders} ; do
  SAMPLE_CSV=$folder/ls_metrics.csv
  for metric in "PRI" "VOI" "Covering" "BF" ; do 
    TGT_CSV=${BASE_SEG_FOLDER}/OIS_${metric}.csv

    ORDER=1
    if [ $metric = "VOI" ] ; then
      ORDER=-1
    fi

    if [ $FIRST -eq 1 ] ; then 
      cp $SAMPLE_CSV $TGT_CSV
    else
      $CSVUPDATE_DEMO $TGT_CSV $SAMPLE_CSV $metric $ORDER 
    fi
  done
  FIRST=0
done

TGT_CSV=${BASE_SEG_FOLDER}/OIS_AGGREGATE.csv
printf "metric," > $TGT_CSV
${METRICS_DEMO} -i dummy -g dummy -p 4 -ls >> $TGT_CSV
printf "\n" >> ${TGT_CSV}
for metric in "PRI" "VOI" "Covering" "BF" ; do 
  METRIC_CSV=${BASE_SEG_FOLDER}/OIS_${metric}.csv
  printf "%s," $metric >> $TGT_CSV
  $AGGREGATE_DEMO $METRIC_CSV 2 >> $TGT_CSV 
  printf "\n" >> $TGT_CSV
done
