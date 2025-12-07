#ifndef _OPFIMG_H_
#define _OPFIMG_H_


#include "cimage.h"
#include "scene.h"
#include "mathematics.h"
#include "adjacency.h"
#include "adjacency3.h"
#include "set.h"
#include "sort.h"
#include "heap.h"
#include "realheap.h"
#include "feature.h"
#include "feature3.h"

#include "subgraph.h"
#include "sgctree.h"

#include "OPF.h"

#include <time.h>


///extern int MAXDENS;   // Maximum value for pdf computation

///#define MAXARCW 1000   // Maximum arc weight for supervised
		       // classification using complete graphs
// Status of the graph nodes for classification

///#define PROTOTYPE   1

// Tie break types
#define NO_TIE 0 // No tiebreak
#define BAYES_TIE 1 // Bayesian tiebreak
#define MEAN_TIE 2 // Mean tiebreak

// Tie map types
#define SCENE_MAP 0 // Distances to the root in the scene space
#define FEATS_MAP 1 // Distance to the root in the features space

//variables used for precomputed distances
///extern bool     PrecomputedDistance;
///extern float  **DistanceValue;
//extern int      MAXDENS;

// Pointer to dist function
//typedef float (*distfun)(float *f1, float *f2, int n);
// redefine dist function
//extern distfun DistFun;
// redefine statistics vectors
// extern float *f1_mean, *f2_mean;
// extern float f1_stddev, f2_stddev;

typedef struct _cnode {
  float  dens;    // density
  float *feat;    // feature vector
  int   *adj;     // list of adjacent nodes
  float *dist;    // distance from adjacent nodes
  int    pixel; // position in the image/scene
} CNode;
//
///typedef struct _snode {
//  float dens;    // probability density value
//  float pathval; // path value
//  float radius; // path value
//  int   label;   // node label
//  int   root;    // root node
//  float volume;  // volume of the cluster
//  int   pred;    // predecessor node
//  int truelabel; // true label if it is known
//  int position;  // index in the image/scene
//  float *feat;    // feature vector
//  Set  *adj;     // list of adjacent nodes
//  char  status;  // 0 - nothing, 1 - prototype
//} SNode;
//
///typedef struct _subgraph {
//  SNode *node;   // nodes of the image/scene subgraph
//  int   nnodes;  // number of nodes
//  int   nfeats;  // number of features
//  int   bestk;   // number of adjacent nodes
//  float entropy; // entropy of the pdf
//  int   nlabels; // number of clusters
//  float df;      // radius in the feature space for density computation
//  float dm;      // radius for markov feature computation
//  float di;      // radius in the image space for density computation
//  float mindens; // minimum density value
//  float maxdens; // maximum density value
////  float K1,K2;  // Constants for PDF computation
//  float K;  // Constant for PDF computation
//  int   Imax;    // Maximum image value;
//  int  *ordered_list_of_nodes; // Store the list of nodes in the increasing/decreasing order of pathval for speeding up classification.
//} Subgraph;

typedef struct _lnode {
  int   size;        // number of voxels in the region
  int   label;       // region label
  float mean;        // brightness mean
  int   rep;         // representant region truelabel
  Set   *adj;        // adjacent regions
  Set   *voxels;     // voxels list
  Curve *histogram;  // voxels normalized and simplified histogram
} LNode;

typedef struct _labelgraph {
  LNode *node;     // nodes of the image/scene subgraph
  int   nnodes;    // number of nodes
  float max_mean;  // maximum brightness mean
} Labelgraph;

// ---- Component tree for pdf filtering ------------
//
///typedef struct _sgctnode {
//  int  level;   /* gray level */
//  int  comp;    /* representative pixel of this node */
//  int  dad;     /* dad node in the maxtree */
//  int *son;     /* son nodes in the maxtree */
//  int  numsons; /* number of sons in the maxtree */
//  int  size;    /* number of pixels of the node */
//} SgCTNode;
//
///typedef struct _sgctree {
//  SgCTNode *node;     /* nodes of the mtree */
//  int      *cmap;     /* component map */
//  int       root;     /* root node of the mtree */
//  int       numnodes; /* number of nodes of the maxtree */
//} SgCTree;



/*----------- Constructor and destructor ------------------------*/

///Subgraph *CreateSubgraph(int nnodes); // Allocate nodes without
				      // feature space and adjacency list
///void DestroySubgraph(Subgraph **sg); // Deallocate memory for subgraph

Labelgraph *CreateLabelgraph( int nnodes ); // Allocate nodes of labelgraph

void DestroyLabelgraph( Labelgraph **lg ); // Deallocate memory for labelgraph

Labelgraph *CopyLabelgraph( Labelgraph *in ); // Copy labelgraph with nodes

/* -----------Labelgraph sets constructors -----------------------*/

void SetLabelSubGraphData( Labelgraph *lg, Subgraph *sg, Scene *scn, Scene *mask );
void SetLabelSubGraphDatabyScene( Labelgraph *lg, Scene *label, Scene *scn, Scene *mask );
                             // Sets histogram, adjacents, and voxels to labelgraph

/*----------- Create subgraph (training set) with image/scene sampling
  ----------- and indicate the image/scene position of each node ----- */

// Get nnodes of samples using uniform random sampling

Subgraph *RandomSampl( Image *img, int nnodes );

// Get samples within a binary mask (optional) by skipping dx pixels
// along x and dy pixels along y. if mask is null, then the image
// domain is used.

Subgraph *UnifSampl(Image *img, Image *mask, int dx, int dy, int xo, int yo);
Subgraph *UnifSampl3(Scene *scn, Scene *mask, int dx, int dy, int dz);

// Compute samples for image compression
Subgraph *ChessSampl(Image *img, int xinit);
Subgraph *VertSampl(Image *img);

// Get samples within a binary mask (optional) by choosing half below
// and half above a given threshold.

Subgraph *ThresSampl3(Scene *scn, Scene *mask, int thres, int nnodes);
Subgraph *ThresNSampl3(Scene *scn, Scene *mask, int *thres, int N, int nnodes);

// Romdom samples inside mask or, if mask is NULL inside image. The
// samples must have similar histogram.

Subgraph *RandomSampl3(Scene *scn, Scene *mask, int nnodes);

// Samples from upper and lower side o histogram divided by threshold thres.

Subgraph *HistogramSampl3(Scene *scn, Scene *mask, float thres , int nnodes );

/*----------- Feature computation based on Markov properties -----------*/

// The intensity/color of a node and of its neighboors within a di
// radius in the image/scene are used to form its feature vector.

void MarkovFeat3(Subgraph *sg, Scene *scn, Scene *mask, float dm);

/*--------- Training functions compute an optimum path forest for
  --------- further classification ----------------------------- */

// Unsupervised training
///void UnsupTrain(Subgraph *sg);

// Supervised training for complete graph
///void SupTrainCompGraph(Subgraph *sg);
// Supervised training for knn-graph
void SupTrainKnnGraph(Subgraph *sg);
// Semi-supervised training for knn-graph
void SemiSupTrainKnnGraph(Subgraph *sg);

/*--- Classification functions ---------------------------------- */
Image *ImagePDF(Subgraph *sg, Features *f);
Image *ImageLabel(Subgraph *sg, Image *img, int vol);
void SceneLabel( Subgraph *sg, Scene *mask );
Image *ImageClassKnnGraph(Subgraph *sg, Features *f);
Scene *SceneClassKnnGraph(Subgraph *sg, Scene *mask, Features3 *f);
Image *ImageClassCompGraph(Subgraph *sg, Features *f);
Scene *SceneCluster(Subgraph *sg, Features3 *f, Scene *mask, int tiebreak);
Scene *FasterSceneCluster(Subgraph *sg, Features3 *f, Scene *mask, int tiebreak);
Scene *FastSceneCluster(Subgraph *sg, Features3 *f, Scene *mask, int tiebreak);
// SceneClusterforMajorityVote returns label map
Scene *SceneClassifyBayes( Subgraph *sg, Features3 *f, Scene *mask );
// SceneCluster returns the label of the classification
Scene **SceneClusterforMajorityVote( Subgraph *sg, Features3 *f, Scene *mask, int tiebreak );

//// Classify nodes of the evaluation/test set for KNNGraph
//void ClassifyKnnGraph(Subgraph *sgtrain, Subgraph *sg);
//// Classify nodes of evaluation/test set for CompGraph
//void ClassifyCompGraph(Subgraph *sgtrain, Subgraph *sg);
//// Classify nodes of evaluation/test set By bayes
void ClassifyBayes(Subgraph *sgTrain, Subgraph *sg);

/*------------ Auxiliary functions ------------------------------ */

// Compute normalized cut
///float NormalizedCut( Subgraph *sg );

// Compute the influence zones of the pdf's maxima
///void UnsupOPF(Subgraph *sg);

// Unsupervised OPF with spatial constrains
void SpatialUnsupOPF3( Subgraph *sg, Scene *scn );
// Compute distance function (listed below) between feature vectors
//float DistFun(float *f1, float *f2, int n);

// Compute Euclidean distance between feature vectors
///float EuclDist(float *f1, float *f2, int n);

// Compute Jaccard distance between feature vectors
float JaccDist(float *f1, float *f2, int n);
// Compute Extended Jaccard distance between feature vectors
float ExtJaccDist(float *f1, float *f2, int n);
// Compute cosine distance between feature vectors
float CosineDist(float *f1, float *f2, int n);
// Compute Pearson's correlation distance between feature vectors
// Warning: must redefine dist_mean and dist_stddev
float CorrelationDist(float *f1, float *f2, int n);

// Compute  chi-squared distance between feature vectors
///float ChiSquaredDist(float *f1, float *f2, int n);
// Compute  Manhattan distance between feature vectors
///float ManhattanDist(float *f1, float *f2, int n);
// Compute  Canberra distance between feature vectors
///float CanberraDist(float *f1, float *f2, int n);
// Compute  Squared Chord between feature vectors
///float SquaredChordDist(float *f1, float *f2, int n);
// Compute  Squared Chi-squared between feature vectors
///float SquaredChiSquaredDist(float *f1, float *f2, int n);
// Compute  Bray Curtis distance between feature vectors
///float BrayCurtisDist(float *f1, float *f2, int n);

// Create arcs for knn subgraph
///void CreateArcs(Subgraph *sg, int knn);

// Create arcs for knn subgraph within the each class
void CreateArcsByTrueLabel(Subgraph *sg, int knn);
// Create symmetric arcs for subgraph
void CreateSymArcs(Subgraph *sg);

// Destroy arcs in knn subgraph
///void DestroyArcs(Subgraph *sg);
// Compute probability density function
///void PDF(Subgraph *sg);

// Compute probability density function for the supervised case
void SupPDF(Subgraph *sg);
// Compute probability density function with space constraint for image
void SPDF(Subgraph *sg, Image *img);
// Compute probability density function with space constraint for scene
void SPDF3( Subgraph *sg, Scene *scn );
// Create classification node
CNode *CreateCNode(Subgraph *sg);
// Destroy classification node
void DestroyCNode(CNode **node);
// Compute Markov features for classification node
void MarkovNodeFeat3(Scene *scn, Scene *mask, CNode *node, int p, float dm, int Imax);
// Copy feature from f to node of position p.
void CopyNodeFeat3( Features3 *f, CNode *node, int p );
// Compute the knn arcs for node
void NodeArcs(Subgraph *sg, CNode *node);
void NodeArcsInSubforest(Subgraph *sg, CNode *node);
void NodeArcsByLabel(Subgraph *sg, CNode *node, int label);
void NodeArcsByTrueLabel(Subgraph *sg, CNode *node, int label);
// Compute node density for classification node
void NodePD(Subgraph *sg, CNode *node);
// Compute node density for classification node by considering a desired label
void NodePDByLabel(Subgraph *sg, CNode *node, int label);
void NodePDByTrueLabel(Subgraph *sg, CNode *node, int label);
// Compute the predecessor in the best path to the classification node
int PredBestPath(Subgraph *sg, CNode *node);
// Compute the value of the best path to the classification node
int ValueBestPath(Subgraph *sg, CNode *node);
// Compute the predecessor in the best path to the classification node using Bayes tie break
int PredBestPathBayesTie(Subgraph *sg, CNode *node);
// Compute the predecessor in the best path to the classification node using mean distance tie break
int PredBestPathMeanTie(Subgraph *sg, CNode *node);
// Find root node and check flatzone
int RootNode(Subgraph *sg, int p, char *flatzone);

///void MSTPrototypes(Subgraph *sg);

// Compute the influnce zones of the maxima using their true labels
void SupOPF(Subgraph *sg);
void SupOPFwithErrors(Subgraph *sg); // allow errors in the training set
// Compute influence zones of prototypes
void SemiSupOPF(Subgraph *sg);
// Set features to the subgraph's nodes
void SetSubgraphFeatures(Subgraph *sg, Features *f);
// Set 3D features to the subgraph's nodes
void SetSubgraphFeatures3(Subgraph *sg, Features3 *f);

// Convert sample distances into integer arc weights
///int ArcWeight(float dist, float maxdist);

/// Copy subgraph (does not copy Arcs)
///Subgraph *CopySubgraph(Subgraph *g);

// Split subgraph into two parts such that the size of the first part
// is given by a percentual of samples.
///void SplitSubgraph(Subgraph *sg, Subgraph **sg1, Subgraph **sg2, float perc1);

// Takes the same amount of samples from subgraph sg and returns a subgraph composed of
// them. The number of samples in given as a parameter.
Subgraph *SubSampleLabeledSubgraph( Subgraph *sg, int nsamples );
// Move errors above max_abs_error from the evaluation set to the
// training set, returning number of errors
int MoveErrorsToTrainingSet(Subgraph **sgtrain, Subgraph **sgeval, int max_abs_error);
// Remove nodes with distinct labels and distance=0
void RemoveInconsistentNodes(Subgraph **sg);
// Merge disjoint subgraphs with no arcs
Subgraph *MergeSubgraphs(Subgraph *sg1, Subgraph *sg2);

//Compute accuracy of classification
///float Accuracy(Subgraph *g);

//Compute accuracy per class
float *AccuracyByLabel(Subgraph *g);

// Copy SNode
///void CopySNode(SNode *dest, SNode *src, int nfeats);
//Swap nodes
///void SwapSNode(SNode *a, SNode *b);
//Replace errors from evaluating set by non prototypes from training set
///void SwapErrorsbyNonPrototypes(Subgraph **sgtrain, Subgraph **sgeval);

//Replace errors from evaluating set by randomly samples from training set
void SwapErrorsbySamples(Subgraph **sgtrain, Subgraph **sgeval);
//Resets subgraph fields (pred)
void ResetCompGraph(Subgraph *sg);

//Executes the learning procedure for CompGraph replacing the
//missclassified samples in the evaluation set by non prototypes from
//training set
///void LearningCompGraph(Subgraph **sgtrain, Subgraph **sgeval, int iterations);

// Baysian learning
void LearningBayes(Subgraph **sgtrain, Subgraph **sgeval, int iterations, int kmax);
//Executes the learning procedure for KnnGraph replacing the
//missclassified samples in the evaluation set by non prototypes from
//training set
void LearningKnnGraph(Subgraph **sgtrain, Subgraph **sgeval, int iterations, int kmax);

// Compute the best k with minimum cut
///void BestkMinCut(Subgraph *sg, int kmax);
// Compute the best k with minimum cut, using kmin parameter
///void BrainBestkMinCut(Subgraph *sg, int kmin, int kmax);

// Estimate the best k by minimum entropy
void ComputeEntropy(Subgraph *sg);

void BestkMinEntropy(Subgraph *sg, int kmax);
// Estimate the best k by Bayes
void BestkMinErrorBayes(Subgraph *sgTrain, Subgraph *sgEval, int kmax);
// Compute the best k with minimum errors
void BestkMinError(Subgraph *sgTrain, Subgraph *sgEval, int kmax);
// Compute the best k with minimum cut for n clusters
void BestkMinCutNClusters(Subgraph *sg, int nclusters, int kmax);

// Eliminate maxima below H
///void ElimMaxBelowH(Subgraph *sg, float H);
// Eliminate maxima below Area
///void ElimMaxBelowArea(Subgraph *sg, int A);
// Eliminate maxima below Volume
///void ElimMaxBelowVolume(Subgraph *sg, int V);
//read subgraph from opf format file
///Subgraph *ReadSubgraph(char *file);
//write subgraph to disk
///void WriteSubgraph(Subgraph *g, char *file);

//write labelnode to disk
void WriteLabelnode(LNode *ln, char *file);
//sort subgraph: order = INCREASING or DECREASING
void SortSubgraphByDensity(Subgraph **cg, char order);
//sort subgraph: order = INCREASING or DECREASING
void SortSubgraph(Subgraph **cg, char order);
/*read the precomputed distances between feature vectors*/
float **ReadDistances(char *fileName);

/*normalize features*/
///void NormalizeFeatures(Subgraph *sg);

/* Correct min and max density for classification */
float *SceneCorrMinMaxPDF(Subgraph *sg, Scene *mask, Features3 *f);
// Select "nclusters" maxima with the largest volumes
Set *SelectLargestDomes(Subgraph *sg, int nclusters);
// Compute influence zones of selected maxima
void OPFByMarkers(Subgraph *sg, Set **S);


////---- Functions for pdf filtering
//
//int       SgAncestor(int *dad, int *cmap, int rq);
//int       SgRepresentative(int *cmap, int p);
//SgCTree *CreateSgMaxTree(Subgraph *g);
//void     DestroySgCTree(SgCTree **ctree);
//int     *SgVolumeOpen(Subgraph *g, int thres);
//int      SgVolumeLevel(SgCTree *ctree, int *level, int i,
//		       int thres, int cumvol);
//int      *SgAreaOpen(Subgraph *g, int thres);
//int       SgAreaLevel(SgCTree *ctree, int *level, int i, int thres);
//void      SgCumSize(SgCTree *ctree, int i);


Subgraph* SplitSubgraphByTrueLabel(Subgraph* sg, int label);

int CountNodesByTrueLabel(Subgraph* sg, int label);


#endif
