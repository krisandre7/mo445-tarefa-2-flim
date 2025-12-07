#!/bin/bash

FOLDER=tmp_slices_X4FWSD

mkdir -p ${FOLDER}
convert $1 _slices%05d.png
mv _slices*.png ${FOLDER} 

iftSlicesToScene -i ${FOLDER} -s .png -o $2

rm -rf ${FOLDER}
