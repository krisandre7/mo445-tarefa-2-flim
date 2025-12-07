#!/bin/sh
ITERS=$1
ACTIVE_ITERS=$2
DATASET_PATH=$3
RESULTS_PREFIX=$4
KMAX_PERC=$5

# Basic input validation
if [ -z $5 ]; then
  echo "Usage: $0 <number of splits> <number of active learning iterations> <dataset> <output_prefix> <kmax_perc> [resume split]"
  exit 1
fi

BIN_SPLIT=iftActiveLearningSplitDataSet
BIN_ACTIVE=iftUnsupOPFActiveLearning
BIN_AGGREGATE=iftAggregateActiveLearningResults
TEMP_DIR=/tmp/.active_learning_experiment${RESULTS_PREFIX}
SPLIT_DATASET_DIR=${TEMP_DIR}/split_datasets
CSV_DIR=${TEMP_DIR}/csv

# Do not reset folders when resuming previous experiments
if [ -z $6 ]; then
  rm -rf $TEMP_DIR
  mkdir -p $TEMP_DIR
  mkdir -p $SPLIT_DATASET_DIR
  mkdir -p $CSV_DIR
fi

make $BIN_SPLIT
make $BIN_ACTIVE
make $BIN_AGGREGATE

idx=0
# Resuming mid experiment
if [ $6 ]; then
  idx=$6
fi

while [ $idx -lt $ITERS ]
do
  echo Split $idx
  TRAIN_PATH=${SPLIT_DATASET_DIR}/TRAIN_${idx}.zip
  TEST_PATH=${SPLIT_DATASET_DIR}/TEST_${idx}.zip
  $BIN_SPLIT $DATASET_PATH $TRAIN_PATH $TEST_PATH
  for activeAlgo in 100 120 200 300
  do
    for classifier in 0 1 2
    do
      # Perform each active learning algorithm with each classifier
      # See iftUnsupOPFActiveLearning --help for algorithm number meaning
      echo Algorithm $activeAlgo Classifier $classifier
      CSV_PATH=${CSV_DIR}/${activeAlgo}_${classifier}_${idx}.csv
      $BIN_ACTIVE -d $TRAIN_PATH -e $TEST_PATH -k $KMAX_PERC -a $activeAlgo -c $classifier -m $ACTIVE_ITERS -w $CSV_PATH
    done
  done
  idx=`expr ${idx} + 1`
done

# Aggregate results
echo Starting result aggregation
for activeAlgo in 100 120 200 300
do
  for classifier in 0 1 2
  do
    CSV_PREFIX=${CSV_DIR}/${activeAlgo}_${classifier}
    RESULT_PATH=${RESULTS_PREFIX}_${activeAlgo}_${classifier}.csv
    $BIN_AGGREGATE $CSV_PREFIX $ITERS $RESULT_PATH
  done
done
