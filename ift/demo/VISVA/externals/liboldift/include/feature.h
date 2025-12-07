#ifndef _FEATURE_H_
#define _FEATURE_H_

#include "common.h"
#include "dimage.h"
#include "cimage.h"

typedef struct _felem {
  float *feat;
} FElem;


typedef struct _features {
  FElem *elem;
  int  nfeats;
  int  nelems;
  int  nrows,ncols;
  int  Imax;
} Features;

/// Allocates memory for features
Features* CreateFeatures(int ncols, int nrows, int nfeats);

/// Deallocates memory for features
void      DestroyFeatures(Features **f);

Features* CopyFeatures(Features* feat);

///Gets the given feature for all image pixels
///Returns a double image with the features
DImage* GetFeature(Features* feats, int index);

int SetFeature(Features* feats, int index, DImage* imagefeats);

///Concatenates 2 features.
///If either features are NULL, or they are of different sizes,
///it will be returned NULL
///IMPORTANT: The features are not deallocated
Features* ConcatFeatures(Features* f1, Features* f2);

/// Used to return the image(Features*) to its original size,
/// after it is power-of-2-sided by the Steerable
/// Pyramid filter. Actually, the image is trimmed until
/// it becomes ncols width and nrows height.
Features* TrimFeatures(Features* feats,int ncols,int nrows);

/// Normalize features
void FNormalizeFeatures(Features *f);

// normalized RGB features
Features* RGBCImageFeats(CImage* cimg);

// using reconstructions
Features *MSImageFeats(Image *img, int nscales);
Features *MSCImageFeats(CImage *cimg, int nscales);
// using convolution with gaussian filters
Features *LMSImageFeats(Image *img, int nscales);
Features *LMSCImageFeats(CImage *cimg, int nscales);
Features *LMSFeats(Features* feats, int nscales);

Features *MarkovImageFeats(Image *img, float dm);
Features *KMarkovImageFeats(Image *img); // for image compression
Features *MarkovCImageFeats(CImage *cimg, float dm);

/// Lab extraction
Features *LabCImageFeats(CImage *cimg);
// This function converts Features* stored in RGB
// to Lab and is generalized for multiple scales.
// Our assumption is that the feature vectors
// in *rgb follow:
// R_1,R_2,...,R_nscales-1,G_1,G_2,...,G_nscales-1,B_1,B_2,...,B_nscales-1
Features *LabFeats(Features *rgb);

/// Shifts Lab negative values
void NormalizeLab(Features *lab);

///Extracts Multi-scale Low-Pass features
Features *MSLowPassFeats(Features* feats, int nscales);

///Extracts Steerable Pyramid Features from the feats
Features *SteerablePyramidFeats(Features* feats, int nscales, int norientations);
///Extracts Steerable Pyramid features from feats and shifts them according the energy
///computed for the scales and orientations
Features *ShiftedSPFeats(Features* feats, int nscales, int norientations);
///Extracts Lab and uses Steerable Pyramid on each band
Features *CLabShiftedSPFeats(CImage* cimg, int nscales, int norientations);
// Convert HSV image to features
Features* HSVCImageFeats(CImage* cimg);


#endif // _FEATURE_H_
