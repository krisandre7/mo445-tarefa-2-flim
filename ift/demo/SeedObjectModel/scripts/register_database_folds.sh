#!/bin/bash


if [ $# -lt 3 ]; then
    echo "usage $0 <original labels folder> <all_images.csv> <database fold 1>  <database fold 2> ... ";
    exit -1;
fi

for i in "${@:3}"; do
    if [ -d $i ]; then
	if [ $i != "param_validation_folds" ]; then
	    python $NEWIFT_DIR/demo/SeedObjectModel/register_database.py --input_all_imgs $2 --input_train_imgs $i/train.csv --input_label_dir $1 --image_depth 12 --affine_params $NEWIFT_DIR/demo/Registration/ElastixConfiguration/Par0000affine.txt --deformable_params $NEWIFT_DIR/demo/Registration/ElastixConfiguration/Par0000bspline.txt --output $i;
	fi
    fi 
done
