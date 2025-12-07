#ifndef _METRICS_H_
#define _METRICS_H_

#include "descriptor.h"
#include "spectrum.h"

double ZNCC(FeatureVector1D *v1, FeatureVector1D *v2);
double FourierDistance(FeatureVector1D *v1, FeatureVector1D *v2);
double EuclideanDistance(FeatureVector1D *v1, FeatureVector1D *v2);
double L1Distance(FeatureVector1D *v1, FeatureVector1D *v2);
double Lifnt(FeatureVector1D *v1, FeatureVector1D *v2);
double KSStatistics(FeatureVector1D *v1, FeatureVector1D *v2);
#endif



