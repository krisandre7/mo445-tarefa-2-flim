#ifndef _CTREE3_H_
#define _CTREE3_H_

#include "scene.h"
#include "adjacency3.h"
#include "gqueue.h"
#include "set.h"
#include "ctree.h"


typedef struct _ctree3 {
  CTNode *node;     /* nodes of the mtree */
  Scene  *cmap;     /* component map */
  int     root;     /* root node of the mtree */
  int     numnodes; /* number of nodes of the maxtree */
} CTree3;

CTree3 *CreateMaxTree3(Scene *scn, AdjRel3 *A);
CTree3 *CreateMinTree3(Scene *scn, AdjRel3 *A);
void   DestroyCTree3(CTree3 **ctree);
Scene *CTAreaClose3(Scene *scn, int thres);
Scene *CTAreaOpen3(Scene *scn, int thres);
Scene *CTVolumeOpen3(Scene *scn, int thres);
Scene *CTVolumeClose3(Scene *scn, int thres);

#endif
