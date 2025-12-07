#ifndef _MORPHOLOGY_H_
#define _MORPHOLOGY_H_

#include "image.h"
#include "annimg.h"
#include "adjacency.h"
#include "gqueue.h"


Image  *Dilate(Image *img, AdjRel *A);
Image  *Erode(Image *img, AdjRel *A);
Image  *Close(Image *img, AdjRel *A);
Image  *Open(Image *img, AdjRel *A);
Image  *MorphGrad(Image *img, AdjRel *A);
Image  *AsfOC(Image *img, AdjRel *A);
Image  *AsfnOC(Image *img, int ntimes);
Image  *AsfOCO(Image *img, AdjRel *A);
Image  *AsfnOCO(Image *img, int ntimes);
Image  *AsfCO(Image *img, AdjRel *A);
Image  *AsfnCO(Image *img, int ntimes);
Image  *AsfCOC(Image *img, AdjRel *A);
Image  *AsfnCOC(Image *img, int ntimes);
Image  *FastAreaOpen(Image *img, int thres);
Image  *FastAreaClose(Image *img, int thres);

/* --------------- IFT-based Operators ------------------------*/

void   iftBasins(AnnImg *aimg, AdjRel *A);
void   iftDomes(AnnImg *aimg, AdjRel *A);
Image *HClose(Image *img, Image *seed, AdjRel *A, int H);
Image *HOpen(Image  *img, Image *seed, AdjRel *A, int H);
//Image *iftSupRec(AnnImg *aimg, AdjRel *A);
Image *iftInfRec(AnnImg *aimg, AdjRel *A);
Image *SupRec(Image *img, Image *marker, AdjRel *A); 
Image *InfRec(Image *img, Image *marker, AdjRel *A); 
Image *SupRecHeap(Image *img, Image *marker, AdjRel *A); /* using a Heap */
Image *RemDomes(Image *img);
Image *CloseHoles(Image *img);
Image *OpenRec(Image *img, AdjRel *A);
Image *CloseRec(Image *img, AdjRel *A);
Image *Leveling(Image *img1, Image *img2);
Image *AsfOCRec(Image *img, AdjRel *A);
Image *AsfnOCRec(Image *img, int ntimes);
Image *AsfOCORec(Image *img, AdjRel *A);
Image *AsfnOCORec(Image *img, int ntimes);
Image *AsfCORec(Image *img, AdjRel *A);
Image *AsfnCORec(Image *img, int ntimes);
Image *AsfCOCRec(Image *img, AdjRel *A);
Image *AsfnCOCRec(Image *img, int ntimes);
Image *AreaClose(Image *img, int thres);
Image *AreaOpen(Image *img, int thres);
Image *LabelAreaOpen(Image *label, int thres);
Image *DilateBin(Image *bin, Set **seed, float radius);
Image *ErodeBin(Image *bin, Set **seed, float radius);
Image *CloseBin(Image *bin, float radius);
Image *OpenBin(Image *bin, float radius);
Image *AsfOCBin(Image *bin, float radius);
Image *FastDilate(Image *img, float radius);
Image  *FeatureGradient(Image *img, int nfeats, int maxval);

#endif


