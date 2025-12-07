#!/bin/bash

if [ $# -ne 1 ]; then
    echo "usage: $0 <input kfold folder>"
    exit -1;
fi

KFOLD_FOLDER=$1
TRAINING_FOLDER=$KFOLD_FOLDER/train
TEST_FOLDER=$KFOLD_FOLDER/test
OUTPUT_SAMPLES_FOLDER=samples
DATASET=$TRAINING_FOLDER/dataset_$OUTPUT_SAMPLES_FOLDER.dat
CLASSIFIER=$TRAINING_FOLDER/opf.dat

### Inverting mask labels
#iftInvertMaskLabels -m $TRAINING_FOLDER/masks
#iftInvertMaskLabels -m $TEST_FOLDER/masks
         
### Splitting training images into tiles
echo "** Splitting training images into tiles **"
iftSplitPlantImagesIntoLabeledTrainingTiles -i $TRAINING_FOLDER/orig -m $TRAINING_FOLDER/masks -c config.json -o $TRAINING_FOLDER/$OUTPUT_SAMPLES_FOLDER

### Computing training dataset from image tiles
echo "** Computing dataset from image tiles **"
iftComputeDatasetFromPlantImageTiles -i $TRAINING_FOLDER/$OUTPUT_SAMPLES_FOLDER -c config.json -o $DATASET

### Training classifier
echo "** Training OPF classifier **"
iftTrainOPFClassifier -i $DATASET -c config.json -o $CLASSIFIER

### Testing a new image
echo "** Classifying test images **"
iftPlantDiseaseLikelihoodMap -i $TEST_FOLDER/orig -c config.json --input-classifier $CLASSIFIER  -o $TEST_FOLDER/results


