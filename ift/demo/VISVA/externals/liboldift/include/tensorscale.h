#ifndef TENSORSCALE_H
#define TENSORSCALE_H 1

#include "common.h"
#include "image.h"
#include "cimage.h"
#include "dimage.h"
#include "geometry.h"
#include "color.h"
#include "adjacency.h"
#include "queue.h"
#include "segmentation.h"
#include "descriptor.h"

#define HISTOGRAMSIZE 180

typedef struct _tensorscale {
  DImage *anisotropy;
  DImage *orientation;
  DImage *thickness;

  int m_pairs;
} TensorScale;

/* Tensor Scale */
TensorScale *CreateBinaryTensorScale(Image *bin, int m_pairs);
void        DestroyTensorScale(TensorScale **ts);
CImage      *ConvertTS2CImage(TensorScale *ts);
void        OutputTSColorSpace(char *filename, int size);

/* Tensor Scale Descriptor (TSD) */
float  *TSOrientationHistogram(TensorScale *ts);
CImage *TSShowHistograms(float *hist1, float *hist2, int offset);
float  TSHistogramMatch(float *hist1, float *hist2, int *offset);

/* Tensor Scale Descriptor with Influence Zones (TSDIZ) */
FeatureVector1D *TSDIZ_ExtractionAlgorithm(Image *bin, int nsamples);
void            WriteTSDIZ(FeatureVector1D *desc,char *filename);
double          TSDIZ_SimilarityAlgorithm(FeatureVector1D *desc1, FeatureVector1D *desc2);

/* Tensor Scale Contour Salience (TSCS) */
FeatureVector2D *TSCS_ExtractionAlgorithm(Image *bin, double threshold); //Salience detector

/* TDE */

Image *TSEDistTrans(Image *bin); // TDE from seed set represented by image bin


#endif
