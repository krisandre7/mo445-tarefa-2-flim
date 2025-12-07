#ifndef ACTIVELEARNINGUTILS_H
#define ACTIVELEARNINGUTILS_H

#include "ift.h"
#include "iftClassification.h"




//RandomSelector
typedef struct ift_random_selector {
  int * nums;
  int totalSize;
  int currentSize; // index for random selection without reposition
} iftRandomSelector;

typedef struct _ift_list_array {
  int size;
  iftList ** list;
} iftListArray;

typedef enum {
  OPFSUP_CLASSIFIER = 0,
  OPFSEMI_CLASSIFIER = 1,
  LOGREG_CLASSIFIER = 2
} iftClassifierAlgorithm;



// Forward declaration for mutual use
struct ift_active_classifier;
typedef struct ift_active_classifier iftActiveClassifier;

// TODO add MST-BE and weighted cluster algorithms
typedef enum {
    ALL_SAMPLES = 0,
    FULL_RANDOM = 1,
    ACTIVE_RWS = 2,
    ACTIVE_RDS = 3
} iftSelectorAlgorithm;

typedef enum {
    FIRST_ITER_DEFAULT = 0,
    FIRST_ITER_RANDOM = 1,
    FIRST_ITER_ROOTS = 2
} iftSelectorFirstIterBehavior;



// In a perfect world this would be an abstract class with each algorithm
//   as a concrete implementation.
// For now, all data stays in this struct with an indication of which
//   algorithms use which data.
typedef struct ift_unlabeled_selector {
    iftSelectorAlgorithm algo;
    iftSelectorFirstIterBehavior firstIterBehavior;
    bool isFirstIter;
    iftDataSet *Z;
    // Algorithm specific data
    iftRandomSelector *rsAll; // FULL_RANDOM
    int * rootArray; // Any based on cluster data
    int nClusters; // Any based on cluster data
    int clusterIndex; // Any based on cluster data
    iftListArray *clusterList; // RWS/RDS
} iftUnlabeledSelector;


typedef struct ift_active_classifier {
  iftClassifierAlgorithm algo;
  bool isTrained;
  iftDataSet * Z;
  iftDataSet * Zlabeled;
  iftDataSet * Zunlabeled;
  iftUnlabeledSelector * semisupSelector;
  // algorithm specific data
  iftCplGraph * OPFClassifier;
  iftLogReg * LogRegClassifier;
} iftActiveClassifier;


// The created selector data must not be changed directly.
// It has a permanent association with _Z_ but does not own it so:
//   - do not free _Z_ while using this
//   - do not expect this to destroy _Z_
// _graph_ is only used in this building phase and can be NULL for some algorithms.
iftUnlabeledSelector * iftCreateUnlabeledSelector(
        iftSelectorAlgorithm algo,
        iftSelectorFirstIterBehavior firstIterBehavior,
        iftDataSet * Z,
        iftKnnGraph * graph);

iftUnlabeledSelector * iftAllocUnlabeledSelector(iftSelectorAlgorithm algo,
                                                 iftSelectorFirstIterBehavior firstIterBehavior,
                                                 iftDataSet * Z);

void iftInitUSAlgorithm(iftUnlabeledSelector * selector, iftKnnGraph * graph);
void iftBuildFullRandomUS(iftUnlabeledSelector * selector);\
void iftBuildActiveRWS(iftUnlabeledSelector * selector, iftKnnGraph *graph);
void iftUSInitClusterData(iftUnlabeledSelector * selector, iftKnnGraph * graph);
void iftBuildActiveRDS(iftUnlabeledSelector * selector, iftKnnGraph *graph);


// Note that that the Unlabeled Selector will override labels from
//   its associated DataSet in some algorithms.
// _nSamples_ is ignored in some cases.
// _classifier_ can be NULL for some algorithms.
iftList * iftPickFromUnlabeledSelector(
    iftUnlabeledSelector * selector,
    int nSamples,
    iftActiveClassifier * classifier);

void iftDestroyUnlabeledSelector(iftUnlabeledSelector ** s);

iftRandomSelector * iftCreateRandomSelector(const int size);
iftListArray * iftCreateListArray(const int arraySize);

iftActiveClassifier * iftCreateActiveClassifier(
    iftClassifierAlgorithm algo,
    iftDataSet * Z,
    iftUnlabeledSelector * semisupSelector);

iftList * iftPickFromUnlabeledSelector(iftUnlabeledSelector * selector,
                                       int nSamples,
                                       iftActiveClassifier * classifier);

iftList * iftPickRootsUS(iftUnlabeledSelector * selector);

iftList * iftPickActiveRDS(iftUnlabeledSelector * selector,
                           int nSamples, iftActiveClassifier * classifier);

int iftRemoveListNode(iftList *L, iftNode **NodeRef, bool isReverse);

void iftActiveClassifierClassifySample(iftActiveClassifier * classifier, iftDataSet * Ztest, int sample);

void iftTrainActiveClassifier(iftActiveClassifier * classifier);

void iftResetActiveClassifierDataSets(iftActiveClassifier * classifier);
bool iftIsSemiSupClassifier(iftActiveClassifier * classifier);
iftDataSet * iftExtractUnlabeledSemisupSamples(iftActiveClassifier * classifier);
void iftResetActiveClassifierClassifiers(iftActiveClassifier * classifier);
void iftTrainActiveClassifierOPFSup(iftActiveClassifier * classifier);
void iftTrainActiveClassifierOPFSemi(iftActiveClassifier * classifier);
int iftActiveClassifierClassifyDataSet(iftActiveClassifier * classifier, iftDataSet * Ztest);
iftList * iftPickFullRandomUS(iftUnlabeledSelector * selector, int nSamples);
void iftResetRandomSelector(iftRandomSelector * rs);
int iftPickFromRandomSelector(iftRandomSelector * rs, const bool hasReposition);
#endif // ACTIVELEARNINGUTILS_H
