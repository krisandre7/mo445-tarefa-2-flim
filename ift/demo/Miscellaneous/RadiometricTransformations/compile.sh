if [ "$#" -ne 1 ]; then
  echo "Usage: sh $0 (0: CPU, 1: GPU)" >&2
  exit 1
fi

make IFT_GPU=$1 iftFindContrastReferenceImage iftStandardizeCTLungs iftMatchHistogram iftStandardizeMRBrain iftMatchHistogramImageSet iftStandardizeMRT1Brain iftMatchHistogramOnMask iftWindowAndLevel iftNormalizeImage iftNormalizeWithNoOutliers iftEqualize iftLinearStretch
