#ifndef _SEGMENTATION_H_
#define _SEGMENTATION_H_

#include "image.h"
#include "cimage.h"
#include "annimg.h"
#include "adjacency.h"
#include "common.h"
#include "curve.h"

Image *GetRoot(AnnImg *aimg);
Image *GetSKIZ(AnnImg *aimg);
Image *GetMarker(AnnImg *aimg);
bool   ValidContPoint(Image *bin, AdjRel *L, AdjRel *R, int p);
Image *LabelContour(Image *bin);
Image *LabelContPixel(Image *bin);
Image *LabelComp(Image *img, AdjRel *A, int thres);
Image *MSLabelComp(Image *img, AdjRel *A, float thres, int nfeats);
Image *LabelBinComp(Image *bin, AdjRel *A);
Image *Paint(Image *bin, int value);
Image *Threshold(Image *img, int lower, int higher);
int    AutoThreshold(Image *img);
int    AcceptEdge(AnnImg *aimg, int *edge);
Image *Clustering(Image *img, AdjRel *A, int Co, int Ao);
Image *Highlight(Image *img, Image *label, int value);
CImage *CHighlight(CImage *cimg, Image *label, float R, float G, float B);
int   *CutPath(AnnImg *aimg, int src, int dst);
Image   *RegionBorder(Image *label);
Image *EdgeDetection(Image *img);

/* -------------- IFT-based Operators ---------------- */

int   *iftFindEdge(AnnImg *aimg, AdjRel *A, int src, int dst);
int   *iftFindOEdge(AnnImg *aimg, AdjRel *A, Image *oimg, char orient, \
		    int src, int dst);
Image *RegMin(Image *img, AdjRel *A);
Image *RegMax(Image *img, AdjRel *A);
Image *HDomes(Image *img, AdjRel *A, int H);
Image *HBasins(Image *img, AdjRel *A, int H);
Image *SelectHDomes(Image *img, Image *seed, AdjRel *A, int H); /* similar to HOpen */
Image *SelectHBasins(Image *img, Image *seed, AdjRel *A, int H); /* similar to HClose */
//Image *iftWatershed(AnnImg *aimg, AdjRel *A);
Image *IncWater(AnnImg *aimg, AdjRel *A);
Image *DecWater(AnnImg *aimg, AdjRel *A);
Image *WaterLabel(Image *img, Image *label, AdjRel *A);
Image *WaterBin(Image *img, Image *bin, AdjRel *A);
Image *WaterGray(Image *img, Image *marker, AdjRel *A);
Image *WaterMax(Image *img, Image *marker, AdjRel *A);

Image *TreeComputing(Image *grad, Curve *pts);
Image *TreeCounting(Image *pred);
Image *TreePruning(Image *nsons, Image *pred, Curve *pts);

void WriteNSons(Image *nsons, Image *pred);
int    Otsu(Image *img);
Image *iftThres(Image *img, AdjRel *A);


#endif



