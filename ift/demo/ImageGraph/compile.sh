if [ "$#" -ne 1 ]; then
  echo "Usage: sh $0 (0: CPU, 1: GPU)" >&2
  exit 1
fi

make IFT_GPU=$1 iftISF_MIX_MEAN iftISF_MIX_MEAN_Simple iftISF_GRID_MEAN iftISF_GRID_ROOT iftISF_MIX_ROOT
