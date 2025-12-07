#ifndef _RADIOMETRIC_H_
#define _RADIOMETRIC_H_

#include "image.h"
#include "curve.h"
#include "gqueue.h"

Curve *Histogram(Image *img);
Curve *NormHistogram(Image *img);
Curve *AccHistogram(Image *img);
Curve *NormAccHistogram(Image *img);
Curve *NormalizeHistogram(Curve *hist);

Image *Probability(Image *img);
Image *LinearStretch(Image *img, int f1, int f2, int g1, int g2);
Image *GaussStretch(Image *img, float mean, float stdev);

void   LinearStretchinplace(Image *img, 
			    int f1, int f2, 
			    int g1, int g2);

// By sorting
Image *Equalize(Image *img, int Imax);
Image *MatchHistogram(Image *img, Image *des);
// Traditional way
Image *TradEqualize(Image *img);
Image *TradMatchHistogram(Image *img, Image *des);

#endif
