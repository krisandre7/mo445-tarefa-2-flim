if [ "$#" -ne 1 ]; then
  echo "Usage: sh $0 (0: CPU, 1: GPU)" >&2
  exit 1
fi

make IFT_GPU=$1 iftBagOfFeatPoints iftCreateLayerModel iftMergeLayerModels iftEncodeLayer iftEncodeMergedLayer iftEncodeMergedLayerBatch iftDecodeLayer iftSortBagOfFeatPoints iftCreateSortedLayerModel iftCreateSortedLayerModelFromBoFP iftCreateMergedLayerModel
