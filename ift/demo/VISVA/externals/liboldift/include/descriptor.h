#ifndef _DESCRIPTOR_H_
#define _DESCRIPTOR_H_

#include "curve.h"
#include "image.h"
#include "cimage.h"
#include "polynom.h"
#include "geometry.h"
#include "common.h"
#include "analysis.h"
#include "adjacency.h"
#include "comptime.h"
#include "segmentation.h"
#include "spectrum.h"

#define LOW 0
#define HIGH 1
#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif

#define DESC_SIZE 64

/*data structure used to compute distances */
typedef struct _FeatureVector1D {
  double *X;
  int n; 
} FeatureVector1D;

typedef struct _FeatureVector2D {
  double *X;
  double *Y;
  int n; 
} FeatureVector2D;

typedef FeatureVector1D *Ap_FeatureVector1D;
typedef FeatureVector2D *Ap_FeatureVector2D;

/* data structure used to extract BIC descriptor */
typedef struct {
  int color;
  int frequency;
}Property;

typedef struct {
  unsigned long colorH[DESC_SIZE];
  unsigned long lowH[DESC_SIZE];
  unsigned long highH[DESC_SIZE];
}VisualFeature;

typedef struct {
  unsigned char colorH[DESC_SIZE];
  unsigned char lowH[DESC_SIZE];
  unsigned char highH[DESC_SIZE];
}CompressedVisualFeature;

FeatureVector1D  *CurveTo1DFeatureVector(Curve *curve);
Ap_FeatureVector2D *Read2DFeatureVectors(char *rfilename, int nimages);
Ap_FeatureVector1D *Read1DFeatureVectors(char *rfilename, int  nimages);
FeatureVector1D *CreateFeatureVector1D(int n);
void            DestroyFeatureVector1D(FeatureVector1D **desc);
void            WriteFeatureVector1D(FeatureVector1D *desc,char *filename);
FeatureVector2D *CreateFeatureVector2D(int n);
void            DestroyFeatureVector2D(FeatureVector2D **desc);
void            WriteFeatureVector2D(FeatureVector2D *desc,char *filename);
void            SortFeatureVector1D(FeatureVector1D *desc, int left, int right, char order);
int             PartFeatureVector1D(FeatureVector1D *desc, int left, int right, char order);
void            SortFeatureVector2D(FeatureVector2D *desc, int left, int right, char order);
int             PartFeatureVector2D(FeatureVector2D *desc, int left, int right, char order);

void Destroy1DFV(Ap_FeatureVector1D **fv, int nimages);

FeatureVector1D *CopyFeatureVector1D(FeatureVector1D *desc);
FeatureVector2D *CopyFeatureVector2D(FeatureVector2D *desc);
void            DescInvertXY(FeatureVector2D *desc);

/**********************************************/
/* Contour Multiscale Fractal Dimension */
Curve *PolynomToFractalCurve(Polynom *P, double lower, double higher, int nbins);
Curve *ContourMSFractal(Image *in);

/**********************************************/
/* Moments Invariant */

Curve *MomentInv(Image *img);
Curve *MomentInvariant(Image *img); // contorno e objeto inteiro
/**********************************************/
/* Fourier Descriptor */

Curve *Image2Curve(Image *img);
Curve *FourierDescriptor(Image *img);
/**********************************************/
/* BIC */

Curve *BIC(CImage *img);
void Write_visual_features(char *filename,char *dbname, CompressedVisualFeature *cvf);
CompressedVisualFeature *Extract_visual_features(CImage *img);
double gray_level_BIC(Image *img1, Image *img2);
#endif
