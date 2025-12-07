#!/bin/bash
DATA_PATH="PATH_TO_IMAGES"
SEEDS_PATH="PATH_TO_SEEDS"
ARCH="PATH_TO_ARCH"
OUTPUT_FOLDER="PATH_TO_SAVE_OUTPUT_FILES"
SALIENCY_PATH="$OUTPUT_FOLDER/saliency" # Save saliencies
OUT_IMAGES_PATH="$OUTPUT_FOLDER/out_images" # Save images with the saliencies' edges drawn
PREDS_PATH="$OUTPUT_FOLDER/preds" # Save predictions and metrics after post-processing (iftSMansoniDelineation)
N_LAYERS=4

# Creates model using Sorted BoFP
python get_encoder.py $DATA_PATH $SEEDS_PATH $ARCH $OUTPUT_FOLDER --merge

for i in $(seq 1 $N_LAYERS);
do
    # Decodes saliency map
    python decode.py $ARCH $OUTPUT_FOLDER $i
    # Gets final saliency map and computes metricsp[]
    iftSMansoniDelineation $SALIENCY_PATH $i $OUT_IMAGES_PATH $PREDS_PATH
done