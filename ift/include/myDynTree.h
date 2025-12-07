#ifndef DYN_TREE_H
#define DYN_TREE_H

#ifdef __cplusplus
extern "C" {
#endif

// #include "ift.h"

/* Forward declaration of struct so we can use it in the function pointer */
typedef struct myDynTrees_ myDynTrees;

/* Define function pointer BEFORE struct uses it */
typedef float (*iftArcWeightFn)(int p, int q, int r, myDynTrees *dt);

typedef struct {
  int   r;      // index of root pixel
  int   label;  // object/background label
  float *mean;  // mean vector (size = m channels)
  int   size;   // current tree size
} CachedRoot;

struct myDynTrees_ {
  iftMImage *img;          /* original image in some color space */
  iftAdjRel *A;            /* adjacency relation */
  iftImage  *label;        /* label map */
  iftImage  *root;         /* root map */
  iftMImage *cumfeat;      /* cumulative feature vector map */
  int       *treesize;     /* tree size map */
  iftImage  *cost;         /* path cost map */
  iftImage  *pred;         /* predecessor map */
  iftGQueue *Q;            /* priority queue with integer costs */
  float      maxfeatdist;  /* maximum feature distance */
  iftArcWeightFn arcwfn;   /* arc-weight function */
  iftFImage *obj;         /* object map */

  // for w2 and w4
  CachedRoot *rootcache;
  int         nroots;

  int **label_sum; // per-label rolling sum of features (for w3 and w6)
  int   *label_size; // per-label rolling count of pixels (for w3 and w6)
  int max_label; // maximum label value (for w3 and w6)

	float *tmp_mean; // temporary buffer for arc weights that need a mean
};

typedef struct ArcWEntry ArcWEntry;

myDynTrees *myCreateDynTrees(iftMImage *img, iftAdjRel *A, iftLabeledSet **S, ArcWEntry *entry);
void         myDestroyDynTrees(myDynTrees **dt);
void         myExecDynTrees(myDynTrees *dt, iftLabeledSet **S);


iftImage *myDynamicTree(iftMImage *img, iftLabeledSet *S, int arcw_id);

//! swig(newobject)
iftImage *myDynamicTreeInOut(iftMImage *img,
                             iftImage *seeds_in,
                             iftImage *seeds_out,
                             int arcw_id);

#ifdef __cplusplus
}
#endif

#endif