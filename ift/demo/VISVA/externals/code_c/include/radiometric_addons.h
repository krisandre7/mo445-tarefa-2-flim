
#ifndef _RADIOMETRIC_ADDONS_H_
#define _RADIOMETRIC_ADDONS_H_

#include "oldift.h"
#include "shared.h"
#include "scene_addons.h"


Curve *XClipHistogram(Curve *hist,
		      int lower, int higher);

Curve *ClipHistogram(Curve *hist, 
		     real left, real right,
		     real bottom, real top);

real GaussianSimilarity(real a, real b, real stdev);
real LinearSimilarity(real a, real b, real dmax);

int    HistogramMedianValue(Curve *hist);

int    AreaPercentageLowerThreshold(Curve *hist,
				    float perc);
int    AreaPercentageHigherThreshold(Curve *hist,
				     float perc);
int    OtsuHistogramThreshold(Curve *hist);
int    NcutHistogramThreshold(Curve *hist, int type);
int    WatershedHistogramThreshold(Curve *hist);
void   SupRec1(int *data, int n, int *cost, int *label, Set *S);


#endif



