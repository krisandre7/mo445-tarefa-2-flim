#!/bin/sh
# 
# PERFORM BENCHMARK ON OISF-CENTER-ISMM ALGORITHM
# --------------------------------------------------------------

# User inputs
DS_NAME="$1"
GAMMA="$2"
TYPE="$3"
CLASSIFIER="$4"

# Dataset paths
IMG_DIR="$DS_DIR/$DS_NAME/orig"
LABEL_DIR="$DS_DIR/$DS_NAME/label"
IDS_PATH="$DS_DIR/$DS_NAME/ids_$TYPE.txt"

# Results paths
RES_DS_DIR="$RES_DIR/$DS_NAME"
RES_SEGM_DIR="$RES_DS_DIR/OISF-CENTER-ISMM-$GAMMA"
RES_SEGM_TYPE_DIR="$RES_SEGM_DIR/$TYPE"

# Temp 
TMP_DIR="$RES_SEGM_TYPE_DIR/tmp"

# Number of superpixels
K_LIST=`seq 10 10 100`

# Validate args
if [ -z "$NEWIFT_DIR" ]; then
  echo "ERROR: \$NEWIFT_DIR is not set! Create \$NEWIFT_DIR which points to IFT dir!"; exit
fi

if [ -z "$DS_DIR" ]; then
  echo "ERROR: \$DS_DIR is not set! Create \$DS_DIR which points to the dataset dir!"; exit
fi

if [ -z "$RES_DIR" ]; then
  echo "ERROR: \$RES_DIR is not set! Create \$RES_DIR which points to the results dir!"; exit
fi

if [ -z "$DS_NAME" ]; then
  echo "ERROR: No dataset name was given! [arg #1]"; exit
elif [ ! -d "$DS_DIR/$DS_NAME" ]; then
  echo "ERROR: Non-existent dataset! <$DS_DIR/$DS_NAME> [arg #1]"; exit
fi

if [ -z "$GAMMA" ]; then
  echo "ERROR: No gamma value was given! [arg #2]"; exit
fi

if [ -z "$TYPE" ]; then
  echo "ERROR: No dataset id type was given! [arg #3]"; exit
elif [ ! -e "$IDS_PATH" ]; then
  echo "ERROR: Non-existent dataset id file! <$IDS_PATH> [arg #3]"; exit
fi

if [ -z "$CLASSIFIER" ]; then
  echo "ERROR: No classifier was given! [arg #4]"; exit
elif [ ! -e "$CLASSIFIER" ]; then
  echo "ERROR: Non-existent classifier! <$CLASSIFIER> [arg #4]"; exit
fi

# Verify programs
if [ -z "$(type -P iftExtrObjSalMap)" ]; then
  echo "ERROR: The program <iftExtrObjSalMap> does not exists \$PATH"; exit
fi

if [ -z "$(type -P iftSeedSampling)" ]; then
  echo "ERROR: The program <iftSeedSampling> does not exists in \$PATH"; exit
fi

if [ -z "$(type -P iftOISFSegm)" ]; then
  echo "ERROR: The program <iftOISFSegm> does not exists \$PATH"; exit
fi

if [ -z "$(type -P iftEvalOverSegm)" ]; then
  echo "ERROR: The program <iftEvalOverSegm> does not exists \$PATH"; exit
fi

if ! [ -e "$NEWIFT_DIR/demo/ObjBasedSuperSegm/python/extrOverallEval.py" ]; then
  echo "ERROR: The program <extrOverallEval.py> does not exists"; exit
fi

# Make directories
if [ ! -d "$RES_DS_DIR" ]; then
  mkdir "$RES_DS_DIR"
fi

if [ ! -d "$RES_SEGM_DIR" ]; then
  mkdir "$RES_SEGM_DIR"
fi

if [ ! -d "$RES_SEGM_TYPE_DIR" ]; then
  mkdir "$RES_SEGM_TYPE_DIR"
fi

if [ -d "$TMP_DIR" ]; then
  rm -rf "$TMP_DIR"
fi
mkdir "$TMP_DIR"

# BEGIN ----------------------------------------------------------------------

# Extract object saliency maps
while IFS='' read -r img_id; do 

  if  [ ! -z "$img_id" ];then  
    IMG_FILE="$(find $IMG_DIR -name $img_id'.*' | head -n1)"
    OBJSM_FILE="$TMP_DIR/objsm_$img_id.png"

    # Extract object saliency maps
    if [ ! -e "$OBJSM_FILE" ]; then
      iftExtrObjSalMap -i "$IMG_FILE" -c "$CLASSIFIER" -o "$OBJSM_FILE" -l 2
    fi
  fi
done < $IDS_PATH

# Run segmentations
for k in $K_LIST; do
  EVAL_FILE="$RES_SEGM_TYPE_DIR/eval_$k.txt"

  if [ ! -e "$EVAL_FILE" ]; then
    echo "Segmenting with $k superpixels -------------------------------------------"
    touch $EVAL_FILE

    # Header
    echo "IMG_ID,BR,UE,OVLAP" >> "$EVAL_FILE"

    while IFS='' read -r img_id; do 

      if  [ ! -z "$img_id" ];then  
        echo "\nIMG ID: $img_id"
        
        # Get img file and its gt
        IMG_FILE=$(find $IMG_DIR -name $img_id'.*' | head -n1)
        GT_FILE=$(find $LABEL_DIR -name $img_id'.*' | head -n1)
        OBJSM_FILE="$TMP_DIR/objsm_$img_id.png"
        SEEDS_FILE="$TMP_DIR/seeds_$img_id.png"
        SEGM_FILE="$TMP_DIR/segm_$img_id.png"

        # Validate
        if [ -z "$IMG_FILE" ]; then
          echo "ERROR: No image with ID $img_id was found!"; exit
        fi

        if [ -z "$GT_FILE" ]; then
          echo "ERROR: No label with ID $img_id was found!"; exit
        fi

        # Sample seeds
        iftCenterSampling -i "$IMG_FILE" -k "$K" -o "$SEEDS_FILE" -sm "$OBJSM_FILE" -or 20.0 -tm 0.2 -ts 0.2

        # Perform segmentation
        iftRunOISF -i "$IMG_FILE" -si "$SEEDS_FILE" -o "$SEGM_FILE" -sm "$OBJSM_FILE" -a 0.5 -b 12 -g "$GAMMA" -it 1 -is 2

        # Evaluate results
        echo "$img_id,$(iftEvalOverSegm -i "$SEGM_FILE" -g "$GT_FILE" -c)" >> "$EVAL_FILE"

        # Remove temporary files
      fi
    done < $IDS_PATH

  else
    echo "Already exists a segmentation file with $k superpixels"
  fi
done

rm -rf "$TMP_DIR"

# Concat files
python $NEWIFT_DIR/demo/ObjBasedSuperSegm/python/extrOverallEval.py -d "$RES_SEGM_TYPE_DIR"

# END ----------------------------------------------------------------------
