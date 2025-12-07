#ifndef _SPECTRUM_H_
#define _SPECTRUM_H_

#include "image.h"
#include "dimage.h"
#include "scene.h"

typedef struct _spectrum {
  double *real;
  double *imag;
  int ncols,nrows,nslices;
  int *tbrow;
  int *tbslice;
} Spectrum;


int       FFT(int dir, long nn, double *x, double *y);
Spectrum *FFT3D(Scene *scn);
Spectrum *FFT2D(Image *img);
Image    *InvFFT2D(Spectrum *spec);
DImage   *DInvFFT2D(Spectrum *spec);
Scene    *InvFFT3D(Spectrum *spec);
Spectrum *CreateSpectrum(int ncols, int nrows);
Spectrum *CreateSpectrum3D(int ncols, int nrows, int nslices);
void      DestroySpectrum(Spectrum **spec);
void      DestroySpectrum3D(Spectrum **spec);
Image    *ViewMagnitude(Spectrum *spec);
Scene    *ViewMagnitude3D(Spectrum *spec);
Image    *ViewPhase(Spectrum *spec);
Scene    *ViewPhase3D(Spectrum *spec);
Spectrum *SpectrumMagnitude(Spectrum *spec);
Spectrum *SpectrumPhase(Spectrum *spec);
Spectrum *SpectrumMagnitude3D(Spectrum *spec);
Spectrum *SpectrumPhase3D(Spectrum *spec);
Spectrum *MultSpectrum(Spectrum *spec1, Spectrum *spec2);
Spectrum *MultSpectrum3D(Spectrum *spec1, Spectrum *spec2);

// Create model of motion degradation due to object movement with
// constant speed (Vx,Vy) during exposure time T.

Spectrum *MotionDegradation(int ncols, int nrows, float Vx, float Vy, float T);


/** FFT2D using DImage as source **/
Spectrum *DFFT2D(DImage *img);

/** Does FFT2D using a real image and an imaginary image
 *  as source. Used in gabor.c because the filter was
 *  not created in frequency domain.
 **/
Spectrum *DFFT2DWithImag(DImage *realimg, DImage* imagimg);

/** Shifts the filter decentering it, so that it can be applied
 *  to the image.
 **/
Spectrum *FFTShift(Spectrum *spec);

/** Shifts the filter using FFTShift and multiplies
 *  the image by it.
 **/
Spectrum* ApplyFilter(Spectrum *imgspec, Spectrum *filter);

Spectrum *CosLowPass(int width, int height, double radius);
Spectrum* SmoothDImageByCosLP(DImage *dimg, double r);

/** Steerable Pyramid **/
Spectrum *CreateSPBandPass(int width, int height, int orientation, double numAngles, double radius);
#endif

