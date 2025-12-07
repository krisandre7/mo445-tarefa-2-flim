#!/bin/bash

path_input=~/Documents/grabcut/orig/*
path_gt=~/Documents/grabcut/labels/
path_res=./

make iftRobotDynamicIFT

if [ -z "$1" ] 
then
	path_res+=robot/
else
	path_res+=$1/
fi

mkdir -p $path_res

for f in $path_input; 
do
	orig_image=$(basename "$f")
	name=${orig_image%.*}
	echo "Processing $name file..."
	iftRobotDynamicIFT -i $f -g $path_gt$name.pgm -m $path_res$name.txt -o $path_res$name.png \
	-a 0.5 -n 90 -t 0.85 -r 2.0 -R 4.0 -N ./data/conv_085_gc_b90.csv
	echo "Done..."
done
