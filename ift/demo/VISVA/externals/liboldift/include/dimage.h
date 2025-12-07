#ifndef _DIMAGE_H_
#define _DIMAGE_H_

//This module is deprecated.
//This file is still available only for
//compatibility purposes.
//You should use "RealMatrix" instead,
//which is far more superior. It supports
//both linear and two-dimensional access
//(M->val[0][p] or M->val[i][j] for a entry
//(i,j) at address p=j+i*ncols) and its
//element values precision can be easily
//changed (float,double,..).

#include "common.h"
#include "image.h"

typedef struct _DImage{
  double *val;
  int ncols,nrows;
  int *tbrow;
} DImage;


DImage *CreateDImage(int ncols, int nrows);
void    DestroyDImage(DImage **dimg);

DImage  *ReadDImage(char *filename);
void    WriteDImage(DImage *dimg, char *filename);
double  MinimumDImageValue(DImage *dimg);
double  MaximumDImageValue(DImage *dimg);
void    SetDImage(DImage *dimg, double value);
bool    ValidDImagePixel(DImage *dimg, int x, int y);
DImage  *CopyDImage(DImage *dimg);
Image   *ConvertDImage2Image(DImage *dimg);

/** ------------- MO445 - 2008 s2 - Project functions -----------*/
DImage 	*ConvertImage2DImage(Image* img);
DImage  *DScale(DImage* img, float Sx, float Sy);

/** Increases the size of DImage until it becomes power-of-2-sided.
 *  The resulting image IS NOT squared.
 **/
DImage* DImagePower2(DImage* img);

/** Shifts image like FFTShift in spectrum.h.
 *  Used mostly in gabor.h.
 **/
DImage* DShift(DImage* img);

#endif
