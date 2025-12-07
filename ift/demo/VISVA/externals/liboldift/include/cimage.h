#ifndef _CIMAGE_H_
#define _CIMAGE_H_

#include "image.h"
#include "radiometric.h"
#include "color.h"
#include "filtering.h"
#include "morphology.h"
#include "mathematics.h"

typedef struct cimage {
  Image *C[3];
} CImage;

CImage *CreateCImage(int ncols, int nrows);
void    DestroyCImage(CImage **cimg);
CImage *ReadCImage(char *filename);
void    WriteCImage(CImage *cimg, char *filename);
CImage *CopyCImage(CImage *cimg);
void    SetCImage(CImage *cimg, int color);
CImage *Convert2CImage(Image *img);
CImage *CLinearStretch(CImage *cimg, int f1, int f2, int g1, int g2);
CImage *CROI(CImage *cimg, int xl, int yl, int xr, int yr);
void    PasteSubCImage(CImage *cimg, CImage *sub, int x, int y);
CImage *CScale(CImage *cimg, float Sx, float Sy);
CImage *CZoom(CImage *cimg, float Sx, float Sy);
CImage *COverlayAlphaImage (CImage *cimg, Image *alpha);
Image  *GetBand(CImage *cimg, char band);
CImage *ColorizeImage(Image *img, float R, float G, float B);
CImage *BlendImages(CImage *cimg1, CImage *cimg2, float alpha1, float alpha2);
Image  *ColorGradient(CImage *cimg);
Image  *CFeatureGradient(CImage *cimg, int maxval);
CImage *CImageRGBtoYCbCr(CImage *cimg);
CImage *CImageYCbCrtoRGB(CImage *cimg);
CImage *CImageRGBtoHSV(CImage *cimg);
CImage *CImageHSVtoRGB(CImage *cimg);
CImage *CImageRGBtoLUV(CImage *rgb);
void    DrawCPoint(CImage *cimg, int x, int y, float raio, int R, int G, int B);
CImage *DrawCBorder(Image *img, Image *label);
void    DrawCImageLineDDA(CImage *cimg, int x1, int y1, int xn, int yn, int color);
CImage *DrawLabeledRegions(Image *img, Image *label);


/** Increases the size of CImage until it becomes power-of-2-sided.
 *  The resulting image IS squared.
 **/
CImage* CImagePower2Squared(CImage* img);

/** Returns 1 if cimg is grayscale, 0 otherwise **/
int IsCImageGray(CImage* cimg);

CImage *CConvertToNbits(CImage *cimg, int N);

CImage* CImageRGBtoLab(CImage* cimg);
CImage* CImageLabtoRGB(CImage* cimg);

CImage* TransformCImage(CImage *img, float M[4][4], int ncols, int nrows);
#endif
