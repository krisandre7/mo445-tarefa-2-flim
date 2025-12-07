#ifndef _BAS_H_
#define _BAS_H_

#include "curve.h"
#include "image.h"
#include "comptime.h"
#include "descriptor.h" //Image2Curve

typedef struct{
  int length;   /* length of the boundary */
  int *X;       /* X values of each boundary point from 0 to length-1   */
  int *Y;       /* Y values of each boundary point from 0 to length-1 */
} boundary_type;

typedef struct{
  int length;   /* length of the boundary  */
  int *mean;    /* mean value BAS function  */
  int *second;  
  int *third;
} representation_type;
/***************************************************/


/*BAS EXTRACTOR*/
/*main file: BAS_ExtractionAlgorithm(Image *in,int rsp,int nsamples)
usage: 	P1: binary image
	P2: resample? rsp = 0 (no) and rsp = 1 (yes)
	P3: nsamples: resample size (0 if rsp = 0)*/

FeatureVector1D *BAS_ExtractionAlgorithm(Image *in,int rsp,int nsamples);
Curve           *BAS(Image *in,int rsp,int nsamples);

/*BAS SIMILARITY*/
double BAS_SimilarityAlgorithm(FeatureVector1D *c1, FeatureVector1D *c2);

#endif

