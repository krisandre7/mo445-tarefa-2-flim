#include <ift.h>
#include "iftUnlabeledSelector.h"

// Convenience
typedef iftUnlabeledSelector iftUS;
typedef iftSelectorAlgorithm iftUS_Algo;
typedef iftSelectorFirstIterBehavior iftUS_FIB;

// ----- Private interface -----
// Building functions
void iftInitUSAlgorithm(iftUS * selector, iftKnnGraph * graph);
void iftInitUSFirstIterBehavior(iftUS * selector, iftKnnGraph * graph);
void iftBuildFullRandomUS(iftUS * selector);
void iftBuildActiveRWS(iftUS * selector, iftKnnGraph * graph);
void iftBuildActiveRDS(iftUS * selector, iftKnnGraph * graph);
// Picking functions
iftList * iftPickAllSamplesUS(iftUS * selector);
iftList * iftPickFullRandomUS(iftUS * selector, int nSamples);
iftList * iftPickActiveRWS(iftUS * selector, int nSamples, iftActiveClassifier * classifier);
iftList * iftPickActiveRDS(iftUS * selector, int nSamples, iftActiveClassifier * classifier);
iftList * iftPickRootsUS(iftUS * selector);

// Helper functions
iftUS * iftAllocUnlabeledSelector(iftUS_Algo algo, iftUS_FIB firstIterBehavior, iftDataSet * Z);
bool iftIsSelectorWithClassifier(iftUS * s);
void iftUSInitClusterData(iftUS * selector, iftKnnGraph * graph);

// Temporary forward declaration
void iftActiveClassifierClassifySample(iftActiveClassifier * classifier, iftDataSet * Ztest, int sample);

// ----- Public functions -----
iftUnlabeledSelector * iftCreateUnlabeledSelector(
    iftUS_Algo algo,
    iftUS_FIB firstIterBehavior,
    iftDataSet * Z,
    iftKnnGraph * graph)
{
  // Sanity test
  if (Z == NULL)
    iftError("Invalid DataSet.", "iftCreateUnlabeledSelector");

  iftUS * selector = iftAllocUnlabeledSelector(algo, firstIterBehavior, Z);
  iftInitUSAlgorithm(selector, graph);
  iftInitUSFirstIterBehavior(selector, graph);
  return selector;
}

iftList * iftPickFromUnlabeledSelector(iftUS * selector, int nSamples, iftActiveClassifier * classifier)
{
  // Sanity test
  if (selector == NULL)
    iftError("Invalid Unlabeled Selector.", "iftPickFromUnlabeledSelector");

  iftList * selectedSamples = NULL;

  if (selector->isFirstIter) {
    selector->isFirstIter = false;
    switch (selector->firstIterBehavior) {
      case FIRST_ITER_RANDOM:
        selectedSamples = iftPickFullRandomUS(selector, nSamples);
        return selectedSamples;
      case FIRST_ITER_ROOTS:
        selectedSamples = iftPickRootsUS(selector);
        return selectedSamples;
      default:
        break;
    };
  }

  switch(selector->algo) {
    case ALL_SAMPLES:
      selectedSamples = iftPickAllSamplesUS(selector);
      break;
    case FULL_RANDOM:
      selectedSamples = iftPickFullRandomUS(selector, nSamples);
      break;
    case ACTIVE_RWS:
      selectedSamples = iftPickActiveRDS(selector, nSamples, classifier);
      break;
    case ACTIVE_RDS:
      selectedSamples = iftPickActiveRDS(selector, nSamples, classifier);
      break;
    default:
      iftError("Invalid active learning algorithm", "iftPickFromUnlabeledSelector");
  };
  return selectedSamples;
}

void iftDestroyUnlabeledSelector(iftUnlabeledSelector ** selector)
{
  if (selector != NULL) {
    iftUS * aux = *selector;
    if (aux != NULL) {
      // Do not destroy Z
      iftDestroyRandomSelector(&(aux->rsAll));
      if (aux->rootArray != NULL)
        free(aux->rootArray);
      iftDestroyListArray(&(aux->clusterList));
    }
    free(aux);
    *selector = NULL;
  }
}

// ---- Private Functions ----

void iftInitUSAlgorithm(iftUS * selector, iftKnnGraph * graph)
{
  switch(selector->algo) {
    case ALL_SAMPLES:
      // Do nothing
      break;
    case FULL_RANDOM:
      iftBuildFullRandomUS(selector);
      break;
    case ACTIVE_RWS:
      iftBuildActiveRWS(selector, graph);
      break;
    case ACTIVE_RDS:
      iftBuildActiveRDS(selector, graph);
      break;
    default:
      iftError("Invalid active learning algorithm", "iftInitUSAlgorithm");
  };
}

void iftInitUSFirstIterBehavior(iftUS * selector, iftKnnGraph * graph)
{
  // If default first iteration behavior is different from main algorithm,
  //   the field is changed to re-direct Pick call in first iteration later.
  // Any algorithm that keeps the FIRST_ITER_DEFAULT value will simply use
  //   the same function regardless of being first iteration or not.
  if (selector->firstIterBehavior == FIRST_ITER_DEFAULT) {
    switch (selector->algo) {
      case ACTIVE_RWS:
      case ACTIVE_RDS:
        selector->firstIterBehavior = FIRST_ITER_ROOTS;
        break;
      default:
        break;
    };
  }

  switch (selector->firstIterBehavior) {
    case FIRST_ITER_RANDOM:
      if (selector->rsAll == NULL)
        iftBuildFullRandomUS(selector);
      break;
    case FIRST_ITER_ROOTS:
      if (selector->rootArray == NULL)
        iftUSInitClusterData(selector, graph);
      break;
    default:
      break;
  };
}

void iftBuildFullRandomUS(iftUS * selector)
{
  iftDataSet * Z = selector->Z;
  selector->rsAll = iftCreateRandomSelector(Z->nsamples);
  for (int s = 0; s < Z->nsamples; ++s)
    selector->rsAll->nums[s] = s;
}

void iftBuildActiveRWS(iftUS * selector, iftKnnGraph *graph)
{
  iftDataSet * Z = selector->Z;

  iftUSInitClusterData(selector, graph);

  // Build each cluster list directly from graph ordered nodes
  iftListArray *la = iftCreateListArray(Z->ngroups);
  for (int n = 0; n < graph->nnodes; ++n) {
    int sample = graph->node[graph->ordered_nodes[n]].sample;
    int cluster = Z->sample[sample].label - 1;
    iftInsertListIntoTail(la->list[cluster], sample);
  }
  selector->clusterList = la; 
}

void iftBuildActiveRDS(iftUS * selector, iftKnnGraph *graph)
{
  iftDataSet * Z = selector->Z;

  iftUSInitClusterData(selector, graph);

  // Find size of each cluster to build arrays
  int * clusterSizes = iftAllocIntArray(Z->ngroups);
  for (int s = 0; s < Z->nsamples; ++s)
    clusterSizes[Z->sample[s].label - 1]++;
  int ** clusterSamples = calloc(Z->ngroups, sizeof(int *));
  float ** clusterDist = calloc(Z->ngroups, sizeof(float *));
  for (int i = 0; i < Z->ngroups; ++i) {
    clusterSamples[i] = iftAllocIntArray(clusterSizes[i]);
    clusterDist[i] = iftAllocFloatArray(clusterSizes[i]);
  }

  // Fill in sample indexes and their distance to the root
  int * clusterIndex = iftAllocIntArray(Z->ngroups);
  for (int n = 0; n < graph->nnodes; ++n) {
    int sample = graph->node[graph->ordered_nodes[n]].sample;
    int cluster = Z->sample[sample].label - 1;
    int idx = clusterIndex[cluster]++;

    clusterSamples[cluster][idx] = sample;
    if (idx == 0) {
      // Root case
      clusterDist[cluster][idx] = 0.0;
    } else {
      int root = clusterSamples[cluster][0];
      if (iftDist == NULL) {
        clusterDist[cluster][idx] = Z->iftArcWeight(Z->sample[sample].feat,
            Z->sample[root].feat, Z->alpha, Z->nfeats); 
      } else {
        clusterDist[cluster][idx] = iftDist->distance_table[sample][root];
      }
    }
  }

  // Sort cluster indexes by distance
  for (int c = 0; c < Z->ngroups; ++c)
    iftFQuickSort(clusterDist[c], clusterSamples[c], 1, clusterSizes[c]-1, IFT_INCREASING);

  // Convert into list for easier use
  iftListArray *la = iftCreateListArray(Z->ngroups);
  for (int c = 0; c < Z->ngroups; ++c) {
    for (int s = 0; s < clusterSizes[c]; ++s)
      iftInsertListIntoTail(la->list[c], clusterSamples[c][s]);
  }
  selector->clusterList = la;

  // Clean up
  for (int c = 0; c < Z->ngroups; ++c) {
    free(clusterDist[c]);
    free(clusterSamples[c]);
  }
  free(clusterDist);
  free(clusterSamples);
  free(clusterSizes);
  free(clusterIndex);
}

iftList * iftPickAllSamplesUS(iftUS * selector)
{
  iftDataSet * Z = selector->Z;
  iftList * selectedSamples = iftCreateList();
  for (int s = 0; s < Z->nsamples; ++s) 
    if (Z->sample[s].status != IFT_TRAIN)
      iftInsertListIntoTail(selectedSamples, s);

  return selectedSamples;
}

iftList * iftPickFullRandomUS(iftUS * selector, int nSamples)
{
  iftDataSet * Z = selector->Z;
  iftRandomSelector * rs = selector->rsAll;
  iftList * selectedSamples = iftCreateList();

  iftResetRandomSelector(rs);
  while (selectedSamples->n < nSamples && rs->currentSize > 0) {
    int sample = iftPickFromRandomSelector(rs, false);
    if (Z->sample[sample].status != IFT_TRAIN)
      iftInsertListIntoTail(selectedSamples, sample);
  }

  return selectedSamples;
}

iftList * iftPickActiveRWS(iftUS * selector, int nSamples, iftActiveClassifier * classifier)
{
  return iftPickActiveRDS(selector, nSamples, classifier);
}

iftList * iftPickActiveRDS(iftUS * selector, int nSamples, iftActiveClassifier * classifier)
{
  iftListArray * la = selector->clusterList;
  iftDataSet * Z = selector->Z;
  int * rootArray = selector->rootArray;
  iftList * selectedSamples = iftCreateList();

  // Init cluster iterator info
  int * backendSteps = iftAllocIntArray(selector->nClusters);
  iftNode ** clusterIter = calloc(selector->nClusters, sizeof(iftNode *));
  for (int c = 0; c < selector->nClusters; ++c)
    clusterIter[c] = la->list[c]->head; 

  int emptyClusterStreak = 0;
  while (selectedSamples->n < nSamples && emptyClusterStreak < selector->nClusters) {
    int cluster = selector->clusterIndex;
    bool sampleFound = false;
    int sample = -1;

    // Try to find sample classified differently from root
    // TODO Make separate function for this step
    while (clusterIter[cluster] != NULL) {
      sample = clusterIter[cluster]->elem;
      if (Z->sample[sample].status == IFT_TRAIN) {
        iftRemoveListNode(la->list[cluster], &(clusterIter[cluster]), false);
        continue;
      }

      // This is important to avoid re-selecting samples
      if (Z->sample[sample].label != 0) {
        clusterIter[cluster] = clusterIter[cluster]->next;
        continue;
      }

      iftActiveClassifierClassifySample(classifier, Z, sample);

      // Compare to root
      if (Z->sample[sample].label == Z->sample[rootArray[cluster]].truelabel) {
        clusterIter[cluster] = clusterIter[cluster]->next;
        continue;
      }

      // Sample is classified differently from root
      sampleFound = true;
      break;
    }

    // Pick from end if necessary and possible
    // TODO Make separate function for this step
    if (!sampleFound) {
      clusterIter[cluster] = la->list[cluster]->tail;
      int i = backendSteps[cluster];
      while (true) {
        // Skip samples already selected in forward phase
        while (clusterIter[cluster] != NULL) {
          sample = clusterIter[cluster]->elem;
          if (Z->sample[sample].label != Z->sample[rootArray[cluster]].truelabel)
            clusterIter[cluster] = clusterIter[cluster]->previous;
          else
            break;
        }

        if (clusterIter[cluster] == NULL)
          break;

        if (i-- > 0)
          clusterIter[cluster] = clusterIter[cluster]->previous;
        else
          break;
      };

      if (clusterIter[cluster] != NULL) {
        sample = clusterIter[cluster]->elem;
        sampleFound = true;
        backendSteps[cluster] += 1;
      }
    }

    emptyClusterStreak += 1;
    if (sampleFound) {
      emptyClusterStreak = 0;
      iftInsertListIntoTail(selectedSamples, sample);
    }

    // Wrap around loop update
    selector->clusterIndex += 1;
    if (selector->clusterIndex == selector->nClusters)
      selector->clusterIndex = 0;
  }

  free(clusterIter);
  free(backendSteps);
  return selectedSamples;
}

iftList * iftPickRootsUS(iftUS * selector)
{
  iftList * selectedSamples = iftCreateList();
  for (int c = 0; c < selector->nClusters; ++c)
    iftInsertListIntoTail(selectedSamples, selector->rootArray[c]);
  return selectedSamples;
}

// Helper
iftUS * iftAllocUnlabeledSelector(iftUS_Algo algo, iftUS_FIB firstIterBehavior, iftDataSet * Z) 
{
  iftUS * selector = calloc(1, sizeof(*selector));
  selector->algo = algo;
  selector->firstIterBehavior = firstIterBehavior;
  selector->isFirstIter = true;
  selector->Z = Z;
  // Algorithm specific data
  selector->rsAll = NULL;
  selector->rootArray = NULL;
  selector->nClusters = 0;
  selector->clusterIndex = -1;
  selector->clusterList = NULL;
  return selector;
}

bool iftIsSelectorWithClassifier(iftUS * selector)
{
  return (selector->algo == ACTIVE_RWS || selector->algo == ACTIVE_RDS);
}

void iftUSInitClusterData(iftUS * selector, iftKnnGraph * graph)
{
  iftDataSet * Z = selector->Z;

  // Sanity test
  if (Z->ngroups <= 0 || graph == NULL)
    iftError("Missing dataset cluster data.", "iftInitClusterData");

  selector->nClusters = Z->ngroups;
  selector->clusterIndex = 0;
  selector->rootArray = iftAllocIntArray(Z->ngroups);

  // Get root data
  for (int c = 0; c < Z->ngroups; ++c)
    selector->rootArray[c] = -1;
  for (int n = 0; n < graph->nnodes; ++n) {
    int sample = graph->node[graph->ordered_nodes[n]].sample;
    int cluster = Z->sample[sample].label - 1;
    if (selector->rootArray[cluster] < 0)
      selector->rootArray[cluster] = sample;
  }
}
