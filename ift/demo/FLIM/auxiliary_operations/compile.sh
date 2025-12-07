if [ "$#" -ne 1 ]; then
  echo "Usage: sh $0 (0: CPU, 1: GPU)" >&2
  exit 1
fi

make IFT_GPU=$1  ift2DGliomaDelineation iftBatchNormalization iftApplyBatchNormalization iftBinarizeActivations iftFLIM-CompareBias iftFLIM-ConvertModel2Bias iftFLIM-SelectKernelsManual iftImageNetTransform iftKernelRelevance iftKernelsByClass iftKernelsByOPF iftLabel2Seeds iftMedianFiltering iftSkipConnection iftSalienceMaps iftSMansoniDelineation iftSMansoniDetection iftInferiorRecOnActiv iftInferiorRecOnSalie iftReformatImagesIntoSameDomain iftResidualLayer iftCorrIntensityVariation iftLabelMarkers iftSplitImageSetByFScore
