#ifndef _IFT_UNLABELED_SELECTOR_H_
#define _IFT_UNLABELED_SELECTOR_H_

// Forward declaration for mutual use
struct ift_active_classifier;
typedef struct ift_active_classifier iftActiveClassifier;

#include <ift.h>
#include "iftRandomSelector.h"
#include "iftActiveClassifier.h"
#include "iftListAppend.h"


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

// Note that that the Unlabeled Selector will override labels from
//   its associated DataSet in some algorithms.
// _nSamples_ is ignored in some cases.
// _classifier_ can be NULL for some algorithms.
iftList * iftPickFromUnlabeledSelector(
    iftUnlabeledSelector * selector,
    int nSamples,
    iftActiveClassifier * classifier);

void iftDestroyUnlabeledSelector(iftUnlabeledSelector ** s);

#endif // _IFT_UNLABELED_SELECTOR_H_
