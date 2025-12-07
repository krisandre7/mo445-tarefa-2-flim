#!/bin/bash
DATA_PATH="PATH_TO_IMAGES"
FLAIR_PATH="$DATA_PATH/t2f" # path to modality folder
SEEDS_PATH="PATH_TO_SEEDS"
ARCH="PATH_TO_ARCH"
OUTPUT_FOLDER="PATH_TO_SAVE_OUTPUT_FILES"
SALIENCY_PATH="$OUTPUT_FOLDER/saliency" # Save saliencies
PREDS_PATH="$OUTPUT_FOLDER/preds" # Save predictions and metrics after post-processing (iftGliomaDelineation)
N_LAYERS=3

# Creates model using Sorted BoFP
python get_encoder.py $FLAIR_PATH $SEEDS_PATH $ARCH $OUTPUT_FOLDER --merge

for i in $(seq 1 $N_LAYERS);
do
    # Decodes saliency map
    python decode.py $ARCH $OUTPUT_FOLDER $i
    # Gets final saliency map and computes metrics
    iftGliomaDelineation $DATA_PATH $SALIENCY_PATH t2f $i $PREDS_PATH
done