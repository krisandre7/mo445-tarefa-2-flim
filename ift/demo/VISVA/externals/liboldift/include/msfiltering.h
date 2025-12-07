#ifndef _MSFILTERING_H_
#define _MSFILTERING_H_

#include "common.h"
#include "spectrum.h"
#include "feature.h"

typedef struct
{
    Spectrum ** m_pSpectralFilterBank;
    Features * m_pFeatures;
    int orientations;
    int scales;
}Filterbank;

void DestroyFilterbank( Filterbank ** );

/** Steerable Pyramid filterbank **/

Filterbank* CreateSPFilterBank(int ncols, int nrows, int scales, int orientations, int nbands);

void ApplySPFilterBank(DImage* img, Filterbank* bank, int startingnfeat);

/** Low-pass filterbank **/

Filterbank* CreateLPFilterBank(int ncols, int nrows, int nscales, int nbands);

void ApplyMSLowPass(DImage* img, Filterbank* bank, int startingnfeat);

#endif // _MSFILTERING_H_
