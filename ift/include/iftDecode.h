#include "ift.h"
#include "iftImage.h"

//! swig(newobject, stable)
iftMImage *iftSimpleAdaptiveDecoder(iftMImage *mimg);

//! swig(newobject, stable)
iftMImage *iftProbabilityBasedAdaptiveDecoder(
  iftMImage *mimg, char *model_dir, char *img_basename, int layer
);

//! swig(newobject, stable)
iftMImage *iftMeanBasedAdaptiveDecoder(
  iftMImage *mimg, char *model_dir, char *img_basename, int layer
);
