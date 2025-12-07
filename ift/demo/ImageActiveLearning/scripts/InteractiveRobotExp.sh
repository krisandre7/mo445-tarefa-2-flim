#!/bin/sh
set -x #echo on
set -e #stop on errors

if [ -z $4 ]; then
  echo "Usage $0 <orig_folder> <gt_folder> <result_path> <finest scale> [2nd finest...]"
  exit 1
fi

ORIG_PATH=$1
GT_PATH=$2
RESULT_PATH=$3

shift 3
SCALES=$@
BUILD_HIERARCHY_SCRIPT="${NEWIFT_DIR}/demo/ImageActiveLearning/scripts/BuildStdHierarchy.sh"

filenames=`find ${ORIG_PATH} -type f -printf "%f\n" | sort`
mkdir -p $RESULT_PATH
CSV_PATH=${RESULT_PATH}/dice_vs_markers.csv
rm -f $CSV_PATH

for file in $filenames ; do
  basename="${file%.*}"
  gt=${GT_FOLDER}/${basename}.pgm

  # Each image has a folder for its results
  file_folder="${RESULT_PATH}/${basename}/"
  rm -r -f $file_folder
  mkdir $file_folder
  
  # Obtain hierarchy
  $BUILD_HIERARCHY_SCRIPT ${ORIG_PATH}/$file ${file_folder}/superpixels $SCALES
  sp_maps=`find ${file_folder} -type f -printf "${file_folder}/%f\n" | sort -r`

  iftHierarchyInteractiveSeg ${ORIG_PATH}/$file ${GT_PATH}/$gt $file_folder $sp_maps >> $CSV_PATH
  printf "\n" >> $CSV_PATH
done

