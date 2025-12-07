#!/bin/sh
set -x #echo on
set -e #stop on errors

if [ -z $6 ]; then
  echo "Usage $0 <train.csv> <val.csv> <base_network> <start_layer (0,1...12)> <pruning_ratio [0,1]> <output_prefix> [<test.csv>]" 
  exit 1
fi

# Example of how to set
# SCRIPT_FOLDER=/home/felipe/dev/trunk/demo/ParasiteHybridClassification/
SCRIPT_FOLDER=${0%pruning_script.sh}

PRUNING_SCRIPT=${SCRIPT_FOLDER}pruningSingleVggLayer.py
RETRAIN_SCRIPT=${SCRIPT_FOLDER}retrainPrunedNetwork.py
TEST_SCRIPT=${SCRIPT_FOLDER}testKerasNetwork.py

TRAIN_SET=$1
VAL_SET=$2
BASE_NETWORK=$3
LAYER=$4
PRUNING_RATIO=$5
OUTPUT_PREFIX=$6
TEST_SET=$7

if [ $LAYER -eq 0 ]; then
  CURRENT_NETWORK=$BASE_NETWORK
else
  CURRENT_NETWORK=${OUTPUT_PREFIX}_layer`expr $LAYER - 1`.h5
fi


# Pruning
while [ $LAYER -le 12 ] ; do
  OUTPUT_NETWORK=${OUTPUT_PREFIX}_layer${LAYER}.h5

  python3 $PRUNING_SCRIPT $TRAIN_SET $VAL_SET $LAYER $PRUNING_RATIO $BASE_NETWORK $CURRENT_NETWORK $OUTPUT_NETWORK  

  if [ $LAYER -ne 0 ]; then
    rm $CURRENT_NETWORK
  fi

  CURRENT_NETWORK=$OUTPUT_NETWORK

  LAYER=`expr $LAYER + 1`
done

# Retrain
OUTPUT_NETWORK=${OUTPUT_PREFIX}_final.h5
python3 $RETRAIN_SCRIPT $TRAIN_SET $VAL_SET $CURRENT_NETWORK $OUTPUT_NETWORK
rm $CURRENT_NETWORK
CURRENT_NETWORK=$OUTPUT_NETWORK

if [ $TEST_SET ]; then
  python3 $TEST_SCRIPT $TEST_SET $CURRENT_NETWORK
fi

