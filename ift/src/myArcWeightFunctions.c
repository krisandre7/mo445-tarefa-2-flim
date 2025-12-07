#include "myArcWeightFunctions.h"
#include "ift.h"
#include "myDynTree.h"

// Squared Euclidean distance between two pixels q and p
static inline float pixelDist(int p, int q, myDynTrees *dt) {
    iftMImage *img = dt->img;
    float dist = 0.0f;
    for (ulong b = 0; b < img->m; b++) {
        float diff = img->val[q][b] - img->val[p][b];
        dist += diff * diff;
    }
    return dist;
}

static inline float distToMean(float *I_q, float *mean, ulong m) {
    float d=0.0f;
    for (ulong b=0; b<m; b++) {
        float diff = I_q[b] - mean[b];
        d += diff*diff;
    }
    return d;
}

float iftArcWeight1(int p, int q, int r, myDynTrees *dt) {
    int *treesize = dt->treesize;

    if (treesize[r] == 0) return 0.0f; // safety

    float inv_size = 1.0f / treesize[r];  // Single division
    float dist = 0.0f;
    
    for (ulong b = 0; b < dt->img->m; b++) {
        float mean = dt->cumfeat->val[r][b] * inv_size;  // Multiply instead
        float diff = dt->img->val[q][b] - mean;
        dist += diff * diff;
    }
    return dist;
}

float iftArcWeight2(int p, int q, int r, myDynTrees *dt) {
    int lbl_p = dt->label->val[p];
    float best = IFT_INFINITY_FLT;
    float *I_q = dt->img->val[q];

    for (int i=0; i<dt->nroots; i++) {
        CachedRoot *cr = &dt->rootcache[i];
        if (cr->label != lbl_p || cr->size == 0) continue;
        float dist = distToMean(I_q, cr->mean, dt->img->m);
        if (dist < best) best = dist;
    }

    return (best == IFT_INFINITY_FLT) ? 0.0f : best;
}

float iftArcWeight3(int p, int q, int r, myDynTrees *dt) {
    int obj_label = dt->label->val[r];  // object label of current root
    float *mean = dt->tmp_mean;         // preallocated temporary buffer
    int count = dt->label_size[obj_label];

    if (count <= 0)
        return 0.0f; // safety

    for (ulong b = 0; b < dt->img->m; b++) {
        mean[b] = (float)dt->label_sum[obj_label][b] / count;
    }

    return distToMean(dt->img->val[q], mean, dt->img->m);
}

float iftArcWeight4(int p, int q, int r, myDynTrees *dt) {
    return iftArcWeight1(p,q,r,dt) + pixelDist(p,q,dt);
}

float iftArcWeight5(int p, int q, int r, myDynTrees *dt) {
    return iftArcWeight2(p,q,r,dt) + pixelDist(p,q,dt);
}

float iftArcWeight6(int p, int q, int r, myDynTrees *dt) {
    return iftArcWeight3(p,q,r,dt) + pixelDist(p,q,dt);
}

static ArcWEntry arcw_table[] = {
    {1,  iftArcWeight1},
    {2,  iftArcWeight2},
    {3,  iftArcWeight3},
    {4,  iftArcWeight4},
    {5,  iftArcWeight5},
    {6,  iftArcWeight6}
};
static int arcw_table_size =
    sizeof(arcw_table) / sizeof(arcw_table[0]);

ArcWEntry* getArcWEntry(int id) {
    for (int i=0; i<arcw_table_size; i++)
        if (arcw_table[i].id == id) return &arcw_table[i];
    return NULL;
}
int getArcWTableSize(void) { return arcw_table_size; }
ArcWEntry* getArcWTable(void) { return arcw_table; }