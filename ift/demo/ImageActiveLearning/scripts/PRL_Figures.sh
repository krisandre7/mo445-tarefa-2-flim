#!/bin/sh
set -x #echo on
set -e #stop on errors

if [ -z $NEWIFT_DIR ]; then
  echo "Error -- Requires the $$NEWIFT_DIR environment variable set to base IFT folder."
  exit 1
fi

if [ -z $4 ]; then
  echo "Usage $0 <result-folder> <plot-prefix> [<result.csv> <name>] n > 0 times"
  exit 1
fi

DEMO_DIR=${NEWIFT_DIR}/demo/ImageActiveLearning
EXEC_PY="python ${DEMO_DIR}/py_scripts/SuperpixelPlot.py"

# General names for axis labels
NSP_NAME="Number of superpixels"
TIME_NAME="Time (ms)"
BR_NAME="Boundary Recall"
UE_NAME="Undersegmentation Error"

#PLOT_CHOICE=bird
#PLOT_CHOICE=liver

# Berk Plots
NSP_RANGE="50 600 0"
TIME_RANGE="0 300 0"
BR_RANGE="0.66 1.0 0" 
UE_RANGE="0.000 0.070 0"


# Bird Plots
if [ $PLOT_CHOICE = "bird" ] ; then
  NSP_RANGE="50 800 0"
  TIME_RANGE="0 300 0"
  BR_RANGE="0.75 1.00 0" 
  UE_RANGE="0.00 0.04 0"
fi

# Liver Plots
if [ $PLOT_CHOICE = "liver" ] ; then
  NSP_RANGE="50 800 0"
  TIME_RANGE="0 300 0"
  BR_RANGE="0.6 1.00 0" 
  UE_RANGE="0.00 0.04 0"
fi

NUM_METHODS=0

# Fill in parameters according to input
TGT_FOLDER=$1
PLOT_PREFIX=$2
shift 2
while [ $1 ]; do
  if [ -z $3 ]; then
    echo "Missing param. Format: <res.csv> <name> <nSp_column>"
  fi
  
  CSV_PATH=$1
  CURVE_NAME=$2
  NSP_COL=1
  COLOR=$NUM_METHODS
  LINESTYLE=$NUM_METHODS
  MARKER=0
  BR_COL=`expr $NSP_COL + 2`
  UE_COL=`expr $NSP_COL + 7`
  BR_PARAMS="$BR_PARAMS $CSV_PATH $CURVE_NAME $COLOR $LINESTYLE $MARKER $NSP_COL $BR_COL"
  UE_PARAMS="$UE_PARAMS $CSV_PATH $CURVE_NAME $COLOR $LINESTYLE $MARKER $NSP_COL $UE_COL"

  NUM_METHODS=`expr $NUM_METHODS + 1`
  shift 2
done

# Create graphs
#${EXEC_PY} ${TGT_FOLDER}/berk_time_plot.pdf "${NSP_NAME}" ${NSP_RANGE} "${TIME_NAME}" ${TIME_RANGE} ${NUM_METHODS} ${TIME_PARAMS}
${EXEC_PY} ${TGT_FOLDER}/${PLOT_PREFIX}_br_plot.pdf "${NSP_NAME}" ${NSP_RANGE} "${BR_NAME}" ${BR_RANGE} ${NUM_METHODS} ${BR_PARAMS}
${EXEC_PY} ${TGT_FOLDER}/${PLOT_PREFIX}_ue_plot.pdf "${NSP_NAME}" ${NSP_RANGE} "${UE_NAME}" ${UE_RANGE} ${NUM_METHODS} ${UE_PARAMS}

