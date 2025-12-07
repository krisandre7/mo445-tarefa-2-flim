/**
* Dynamic and Iterative Spanning Forest
* 
* @date September, 2019
*/
#ifndef DISF_H
#define DISF_H

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Includes
//=============================================================================
#include "iftMImage.h"
#include "iftImage.h"
#include "ift/core/dtypes/DHeap.h"
#include "ift/core/dtypes/Set.h"
#include "iftAdjacency.h"


//=============================================================================
// Structures
//=============================================================================
typedef struct
{
    int root_index, num_nodes, num_feats;
    float *sum_feat;
} Tree;

//=============================================================================
// Prototypes
//=============================================================================
Tree *createTree(int root_index, int num_feats); // root note is not inserted
void freeTree(Tree **tree);

double euclDistance(float *feat1, float *feat2, int num_feats); // L2-norm
double taxicabDistance(float *feat1, float *feat2, int num_feats); // L1-norm

float* meanTreeFeatVector(Tree *tree);

double *computeGradient(iftMImage *graph);

// If border_img is not desired, simply pass NULL.
iftImage *runDISF(iftMImage *mimg, 
                int n_0, int n_f, 
                iftAdjRel *A,
                iftImage *mask);

iftSet *selectKMostRelevantSeeds(Tree **trees, iftSet **tree_adj, int num_nodes, int num_trees, int num_maintain);

void insertNodeInTree(iftMImage *graph, int index, Tree **tree);


#ifdef __cplusplus
}
#endif

#endif // DISF_H