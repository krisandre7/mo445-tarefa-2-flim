#ifndef _ANNIMG_H_
#define _ANNIMG_H_

#include "image.h"
#include "scene.h"
#include "adjacency.h"
#include "set.h"

typedef struct _annimg {
  Image *img;
  Image *grad;
  Image *cost;
  Image *label;
  Image *pred;
  Image *root;
  Set   *seed;
} AnnImg;


AnnImg *CreateAnnImg(Image *img);
void    DestroyAnnImg(AnnImg **aimg);
AnnImg *Annotate(Image *img, Image *cost, Image *label);
void   DeAnnotate(AnnImg **aimg);
void   AddSeed(AnnImg *aimg, int pixel, int cost, int label, int pred);
bool   RemSeed(AnnImg *aimg, AdjRel *A, int pixel); /* adj. relation must be the same used to propagate the seed pixel */
Image *CompPaths(Image *pred);
int    Seed(Image *pred, int p);
int    SeedComp(Image *pred, int p);
int   *Path(Image *pred, int dst);
Image *GetCost(AnnImg *aimg);
Image *GetLabel(AnnImg *aimg);
Image *GetPred(AnnImg *aimg);
Image *Label2Root(Image *label);
Image *Root2Label(Image *root, Image *rootlabel);

#endif
