#ifndef _SGCTREE_H_
#define _SGCTREE_H_

#include "common.h"
#include "gqueue.h"
#include "ctree.h"
#include "subgraph.h"

// ---- Component tree for built from Subgraph ------------
typedef struct _sgctree {
  CTNode *node;     /* nodes of the mtree */
  int      *cmap;     /* component map */
  int       root;     /* root node of the mtree */
  int       numnodes; /* number of nodes of the maxtree */
} SgCTree;


SgCTree *CreateSgMaxTree(Subgraph *g);
void DestroySgCTree(SgCTree **ctree);
int SgAncestor(int *dad, int *cmap, int rq);
int SgRepresentative(int *cmap, int p);
void SgCumSize(SgCTree *ctree, int i);
int *SgAreaOpen(Subgraph *g, int thres);
int SgAreaLevel(SgCTree *ctree, int *level, int i, int thres);
int *SgVolumeOpen(Subgraph *g, int thres);
int SgVolumeLevel(SgCTree *ctree, int *level, int i, int thres, int cumvol);

#endif
