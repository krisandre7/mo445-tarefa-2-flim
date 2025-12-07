#ifndef _IFT_ACTIVE_CLASSIFIER_H_
#define _IFT_ACTIVE_CLASSIFIER_H_

// Forward declaration for mutual use
struct ift_unlabeled_selector;
typedef struct ift_unlabeled_selector iftUnlabeledSelector;

#include <ift.h>
#include "iftUnlabeledSelector.h"

typedef enum {
  OPFSUP_CLASSIFIER = 0,
  OPFSEMI_CLASSIFIER = 1,
  LOGREG_CLASSIFIER = 2
} iftClassifierAlgorithm;

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

iftActiveClassifier * iftCreateActiveClassifier(iftClassifierAlgorithm algo, iftDataSet * Z, iftUnlabeledSelector * semisupSelector);
void iftTrainActiveClassifier(iftActiveClassifier * classifier);
void iftActiveClassifierClassifySample(iftActiveClassifier * classifier, iftDataSet * Ztest, int sample);
void iftActiveClassifierClassifyDataSet(iftActiveClassifier * classifier, iftDataSet * Ztest);
void iftDestroyActiveClassifier(iftActiveClassifier ** classifier);

#endif // _IFT_ACTIVE_CLASSIFIER_H_
