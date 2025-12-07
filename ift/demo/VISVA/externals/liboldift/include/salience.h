#ifndef _SALIENCE_H_
#define _SALIENCE_H_

#include "curve.h"
#include "image.h"
#include "analysis.h"
#include "comptime.h"
#include "segmentation.h"
#include "descriptor.h"
#include "metrics.h"

/**********************************************/
/* Contour Saliences */
Curve3D *iftContourSaliences(Image *bin,int threshold_in,int threshold_out,int angle_in,int angle_out);
Curve3D *iftContourConvexSaliences(Image *bin,int threshold,int angle);
Curve3D *iftContourConcaveSaliences(Image *bin,int threshold,int angle);
Curve   *ContourSaliences(Image *in);
Image   *PaintContourSaliences(Image *bin, int threshold_in,int threshold_out,int angle_in,int angle_out, int side);
double  MatchDistance(FeatureVector1D *v1, FeatureVector1D *v2);
double  ContourSalienceMatching(FeatureVector2D *d1, FeatureVector2D *d2);
double  FourierSalienceDistance(FeatureVector2D *d1, FeatureVector2D *d2);

#endif
