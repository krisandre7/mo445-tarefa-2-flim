#ifndef _ANNSCN_H_
#define _ANNSCN_H_

#include "scene.h"
#include "adjacency3.h"
#include "set.h"

typedef struct _annscn {
  Scene *scn;
  Scene *cost;
  Scene *pred;  
  Scene *label;
  Scene *root;
  Set   *seed;
  Set   *mark;
} AnnScn;

Scene  *CreateMapScene(Scene *mainscn);
void    DestroyMapScene(Scene *scn);
AnnScn *Annotate3(Scene *scn);
void    DeAnnotate3(AnnScn **ascn);
void    AddMark3(AnnScn *ascn, int x, int y, int z);
void    AddSeed3(AnnScn *ascn, int x, int y, int z, int cost, int label, int pred);
int     FindRoot(Scene *pred, int t);
int     SeedComp3(Scene *pred, int p);
Scene *GetLabel3(AnnScn *ascn);
Scene *GetCost3(AnnScn *ascn);
Scene *GetPred3(AnnScn *ascn);

/* new annotated scene, (compact annotated scene),
   should replace the old one, but both will coexist 
   for a little */

typedef struct _cvoxel {
  int value;
  int cost;
  int pred;
  int root;
  int label;
} CVoxel;

typedef struct _seedprop {
  int cost, label, pred;
} SeedProp;

typedef struct _cannscn {
  int   size[3];
  int   N, WxH;
  float dx,dy,dz;

  int *tbrow, *tbframe;
  CVoxel *vd;
  
  IntSet *seed; /* M_i */
  IntSet *mark; /* M_r */

} CAnnScn;

#endif
