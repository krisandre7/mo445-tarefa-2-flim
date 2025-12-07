#ifndef _BRAINCLUSTER_OPF_H_
#define _BRAINCLUSTER_OPF_H_

#define BCMAXARCW 100000.0
#define BCMAXDENS 1000.0

typedef float ( *BCArcWeightFun )( float *f1, float *f2, int n );

extern BCArcWeightFun BCArcWeight;

#if _WIN64 || __amd64__ || amd64 || __x86_64__ || x86_64 || ia64 || __ia64__ ||__sparc64__ || _LP64 || __WORDIZE == 64  
typedef long long pointer_type;
#else
typedef int pointer_type;
#endif

typedef struct {
  float *feat;
  float **nodefeat;
  int nfeats;
  int nelems;
  int Imax;
} BCFeatures;

typedef struct {
  float dens;    /* probability density value */
  float radius;  /* maximum distance among the k-nearest neighbors in the training set. It is used to propagate clustering labels to testing nodes) */
  int   label;   /* node label */
  int   pred;    /* predecessor node */
  int   pixel;  /* pixel index in the image/scene */
  int   nplatadj; /*holds the amount of adjacent nodes on plateaus. */
  int  *adj;     /* list of adjacent nodes */
  int  *plt;     /* list of adjacent nodes in plateau */
  float *feat;   /* feature vector */
} BCNode;

typedef struct {
  BCNode *node;  /* nodes of the image/scene subgraph */
  float *feat;   /* feature vectors */
  int   nnodes;  /* number of nodes */
  int   nfeats;  /* number of features */
  int   bestk;   /* number of adjacent nodes */
  int   nlabels; /* number of clusters */
  float df;      /* radius in the feature space for density computation */
  float di;      /* radius in the image space for density computation */
  float mindens; /* minimum density value */
  float maxdens; /* maximum density value */
  float K;  /* Constant for PDF computation */
  int   Imax;    /* Maximum image value */
  int  *ordered_list_of_nodes; /* Store the list of nodes in the increasing/decreasing order of pathval for speeding up classification. */
} BCSubgraph;

/* BCSubgraph functions. */
BCSubgraph *CreateBCSubgraph( int nnodes );
BCSubgraph *BCRandomSampl3( Scene *scn, Scene *mask, int nnodes );
BCSubgraph *BCUnifSampl3( Scene *scn, Scene *mask, int dx, int dy, int dz );
void FreeSubgraph( BCSubgraph **sg );

/* OPF related functions. */
float BCEuclDist( float *f1, float *f2, int n );
float BCEuclDistLog( float *f1, float *f2, int n );
float* BCCreateArcs( BCSubgraph *sg, int kmax );
void BCPDF( BCSubgraph *sg );
void BCSPDF( BCSubgraph *sg, Scene *mask, int first_voxel, int last_voxel );
void BCOPFClustering( BCSubgraph *sg );
float BCNormalizedCutToKmax( BCSubgraph *sg );
void BCRemovePlateauNeighbors( BCSubgraph *sg );
void BCSpatialOPFClustering( BCSubgraph *sg, Scene *mask );
Scene *SceneClassifyKnnGraph( BCSubgraph *sg, Scene *mask, int first_voxel, int last_voxel, BCFeatures *f, int t_threads );
void BCDestroyArcs( BCSubgraph *sg );
void BCBestkClustering( BCSubgraph *sg, int kmin, int kmax );

/* BCFeatures functions. */
BCFeatures* CreateBCFeatures( Scene *scn, Scene *mask, int nfeats );
void DestroyBCFeatures( BCFeatures **f );
void WriteFeats( BCFeatures* feats, Scene *mask, char name[ ] );
BCFeatures *MedianSceneFeatures( Scene *scn, Scene *mask, int first_voxel, int last_voxel, float r, int t_threads );
BCFeatures *MedianSceneFeaturesAtlas( Scene *scn, Scene *mask, Scene *dark_atlas, Scene *light_atlas, int first_voxel, int last_voxel, float r, int t_threads );
BCFeatures *CoOccurFeatures( Scene *scn, Scene *mask, int first_voxel, int last_voxel, int nfeats, int t_threads );
BCFeatures *CoOccurFeaturesAtlas( Scene *scn, Scene *mask, Scene *dark_atlas, Scene *light_atlas, int first_voxel, int last_voxel, int nfeats, int t_threads );
BCFeatures *AdaptiveSceneFeatures( Scene *scn, Scene *mask, int first_voxel, int last_voxel, int protocol, int tissue, float r, int t_threads );
BCFeatures *AtlasFeatures( Scene *scn, Scene *mask, Scene *dark_atlas, Scene *light_atlas, int first_voxel, int last_voxel );
void SetBCFeatures( BCSubgraph *sg, BCFeatures *f );
#endif
