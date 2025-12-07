#!/bin/bash

NUMBER_ITERATIONS="30"
NUMBER_COLORS="40"
NUMBER_SUPERPIXELS="400"
DATASET_NAME=$2
DATASET_PATH=$1


if [ $# -eq 0 ]
  then
    echo "Please specify Dataset path and name 
   runOnDataset <../path-to-datasets/>, <dataset-name>"
    exit
fi

if [ $# -eq 1 ]
  then
    echo "Please specify Dataset name and Path 
   runOnDataset <../path-to-datasets/>, <..dataset-name>"
    exit
fi

for file in "$DATASET_PATH""$DATASET_NAME"/images/*
do
  ./iftIFCS "${file##*/}" "$NUMBER_ITERATIONS" "$DATASET_PATH" "$DATASET_NAME" "$NUMBER_COLORS" "$NUMBER_SUPERPIXELS"
done
