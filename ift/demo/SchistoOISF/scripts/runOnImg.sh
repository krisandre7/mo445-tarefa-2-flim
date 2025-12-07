#!/bin/sh
# 
# RUN OISF-CENTER ALGORITHM ON IMAGE
# --------------------------------------------------------------

# User inputs
IMG_FILE="$1"
OBJSM_FILE="$2"
K="$3"
GAMMA="$4"

# Output files
SEEDS_FILE="seeds.png"
SEGM_FILE="segm.png"
OVLAY_FILE="ovlay.png"
OVSEED_FILE="ovseed.png"

# Validate args
if [ -z "$IMG_FILE" ]; then
  echo "ERROR: No image was given! [arg #1]"; exit
elif [ ! -e "$IMG_FILE" ]; then
  echo "ERROR: Non-existent image! <$IMG_FILE> [arg #1]"; exit
fi

if [ -z "$OBJSM_FILE" ]; then
  echo "ERROR: No object saliency map image was given! [arg #2]"; exit
elif [ ! -e "$IMG_FILE" ]; then
  echo "ERROR: Non-existent object saliency map image! <$OBJSM_FILE> [arg #2]"; exit
fi

if [ -z "$K" ]; then
  echo "ERROR: Inexistent amount of superpixels! [arg #3]"; exit
fi

if [ -z "$GAMMA" ]; then
  echo "ERROR: Inexistent object saliency map confidence! [arg #4]"; exit
fi

# Verify programs
if [ -z "$(type -P iftSeedSampling)" ]; then
  echo "ERROR: The program <iftSeedSampling> does not exists in \$PATH"; exit
fi

if [ -z "$(type -P iftOISFSegm)" ]; then
  echo "ERROR: The program <iftOISFSegm> does not exists \$PATH"; exit
fi

if [ -z "$(type -P iftWriteBorders)" ]; then
  echo "ERROR: The program <iftWriteBorders> does not exists \$PATH"; exit
fi

# BEGIN ----------------------------------------------------------------------
# Sample seeds
echo "Sampling Seeds"
iftCenterSampling -i "$IMG_FILE" -k "$K" -o "$SEEDS_FILE" -sm "$OBJSM_FILE" -or 20.0 -tm 0.2 -ts 0.2

# Perform segmentation
echo "Running OISF"
iftRunOISF -i "$IMG_FILE" -si "$SEEDS_FILE" -o "$SEGM_FILE" -sm "$OBJSM_FILE" -a 0.5 -b 12 -g "$GAMMA" -it 1 -is 2

# Overlay borders
echo "Writing results"
iftWriteBorders -i "$IMG_FILE" -s "$SEGM_FILE" -o "$OVLAY_FILE" -t 1 -cr 0 -cb 1 -cg 1
iftWriteBorders -i "$IMG_FILE" -s "$SEEDS_FILE" -o "$OVSEED_FILE" -t 3 -cr 1 -cb 0 -cg 0

# END ----------------------------------------------------------------------
