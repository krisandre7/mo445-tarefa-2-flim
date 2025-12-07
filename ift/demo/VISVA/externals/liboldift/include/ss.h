#ifndef _SS_H_
#define _SS_H_

#include "curve.h"
#include "adjacency.h"
#include "annimg.h"
#include "segmentation.h"
#include "analysis.h"
#include "descriptor.h"

/*SS EXTRACTOR*/
Curve *SS_ExtractionAlgorithm(Image *in, int maxdist, int nsamples, int side);

/*SS SIMILARITY*/
double SS_getMin(double Dist1, double Dist2, double Dist3);
double SS_OCS(FeatureVector1D *fv1, FeatureVector1D *fv2);
double SS_OCSMatching(FeatureVector1D *fv_1, FeatureVector1D *fv_2);
double SS_SimilarityAlgorithm(FeatureVector1D *c1, FeatureVector1D *c2);

#endif
