

#ifndef _METRICAS_H_
#define _METRICAS_H_

extern "C" {
#include "oldift.h"
#include "simetria.h"
}

/* O Quadrado da Diferença */
#ifndef QD
#define QD(x, y) (x - y)*(x - y)
#endif

#ifndef LEFT_RIGHT
#define LEFT -1
#define RIGHT 1
#endif

typedef struct _hemispheres {
  Scene *left;
  Scene *right;
} Hemispheres;


Image * iftDilation2D(Image *border, AdjRel *A);

Polynom *MSFractal2D(Image *bin, int maxdist, int degree, double lower, 
                     double higher, int reg, double from, double to);

Curve *ContourMSFractal2D(Image *in, int nbins);

Scene * GetHemisphere(Scene *bin, int side, DVoxel normal, DVoxel center,
                      double xshift);

Hemispheres * GetHemispheres(Scene *bin, DVoxel normal, DVoxel center,
                             double xshift);

void SumMetric(Scene *bin);

void HemispheresVolume(Scene *scn);

Scene * iftDilation3D(Scene *border, AdjRel3 *A);

Polynom *MSFractal3D(Scene *border, int maxdist, int degree, double lower, 
                     double higher, int reg, double from, double to);

Curve * ContourMSFractal3D(Scene *bin);

Scene *BinarizeVolume(Scene *scn);

Image *BinarizeImage(Image *img);

Curve * DiffCurves(Curve *curve1, Curve *curve2, double factor);

Curve * StandardDeviationCurve(Curve **curves, Curve *mcurve, int ncurves);

Curve *MedianCurve(Curve **curves, int ncurves);

Curve * HemispheresMedianCurve(Scene *bin);

Curve * Fractal2D(char *basename, Scene *scn);

void Fractal3D(char *basename);

#endif
