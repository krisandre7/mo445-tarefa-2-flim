#ifndef _CTREE_H_
#define _CTREE_H_

#include "annimg.h"
#include "gqueue.h"
#include "set.h"

typedef struct _ctnode {
  int  level;   /* gray level */
  int  comp;    /* representative pixel of this node */
  int  dad;     /* dad node in the maxtree */
  int *son;     /* son nodes in the maxtree */
  int  numsons; /* number of sons in the maxtree */
  int  size;    /* number of pixels of the node */
} CTNode;

typedef struct _ctree {
  CTNode *node;     /* nodes of the mtree */
  Image  *cmap;     /* component map */
  int     root;     /* root node of the mtree */
  int     numnodes; /* number of nodes of the maxtree */
} CTree;

CTree *CreateMaxTree(Image *img, AdjRel *A);
CTree *CreateMinTree(Image *img, AdjRel *A);
void   DestroyCTree(CTree **ctree);
AnnImg *TopoWater(Image *img, AdjRel *A, int flag);
Image *CTAreaClose(Image *img, int thres);
Image *CTAreaOpen(Image *img, int thres);
Image *CTVolumeOpen(Image *img, int thres);
Image *CTVolumeClose(Image *img, int thres);

#endif
