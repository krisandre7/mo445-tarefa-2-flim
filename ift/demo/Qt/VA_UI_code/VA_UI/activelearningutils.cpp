#include "activelearningutils.h"

// Convenience
typedef iftUnlabeledSelector iftUS;
typedef iftSelectorAlgorithm iftUS_Algo;
typedef iftSelectorFirstIterBehavior iftUS_FIB;




iftUnlabeledSelector * iftCreateUnlabeledSelector(
        iftUS_Algo algo,
        iftSelectorFirstIterBehavior firstIterBehavior,
        iftDataSet * Z,
        iftKnnGraph * graph)
{
    // Sanity test
    if (Z == NULL)
        iftError("Invalid DataSet.", "iftCreateUnlabeledSelector");

    iftUnlabeledSelector * selector = iftAllocUnlabeledSelector(algo, firstIterBehavior, Z);
    iftInitUSAlgorithm(selector, graph);
    //iftInitUSFirstIterBehavior(selector, graph);
    return selector;
}

// Helper
iftUnlabeledSelector * iftAllocUnlabeledSelector(iftSelectorAlgorithm algo,
                                                 iftSelectorFirstIterBehavior firstIterBehavior,
                                                 iftDataSet * Z)

{
    iftUnlabeledSelector* selector = (iftUnlabeledSelector *)calloc(1, sizeof(*selector));
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

void iftInitUSAlgorithm(iftUnlabeledSelector * selector, iftKnnGraph * graph){
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

void iftBuildFullRandomUS(iftUnlabeledSelector * selector){
    iftDataSet * Z = selector->Z;
    selector->rsAll = iftCreateRandomSelector(Z->nsamples);
    for (int s = 0; s < Z->nsamples; ++s)
        selector->rsAll->nums[s] = s;
}

void iftBuildActiveRWS(iftUnlabeledSelector * selector, iftKnnGraph *graph){
    iftDataSet * Z = selector->Z;

    iftUSInitClusterData(selector, graph);

    // Build each cluster list directly from graph ordered nodes
    iftListArray *la = iftCreateListArray(Z->nlabels);
    for (int n = 0; n < graph->nnodes; ++n) {
        int sample = graph->node[graph->ordered_nodes[n]].sample;
        int cluster = Z->sample[sample].label - 1;
        iftInsertListIntoTail(la->list[cluster], sample);
    }

    selector->clusterList = la;
}

iftRandomSelector * iftCreateRandomSelector(const int size){
    iftRandomSelector * rs = (iftRandomSelector *)calloc(1, sizeof(iftRandomSelector));
    rs->nums = iftAllocIntArray(size);
    rs->currentSize = rs->totalSize = size;
    return rs;
}

void iftUSInitClusterData(iftUS * selector, iftKnnGraph * graph){
    iftDataSet * Z = selector->Z;

    // Sanity test
    if (Z->nlabels <= 0 || graph == NULL)
        iftError("Missing dataset cluster data.", "iftInitClusterData");

    selector->nClusters = Z->nlabels;
    selector->clusterIndex = 0;
    selector->rootArray = iftAllocIntArray(Z->nlabels);

    // Get root data
    for (int c = 0; c < Z->nlabels; ++c)
        selector->rootArray[c] = -1;
    for (int n = 0; n < graph->nnodes; ++n) {
        int sample = graph->node[graph->ordered_nodes[n]].sample;
        int cluster = Z->sample[sample].label - 1;
        if (selector->rootArray[cluster] < 0)
            selector->rootArray[cluster] = sample;
    }
}

iftListArray * iftCreateListArray(const int arraySize)
{
    iftListArray * la = (iftListArray *)calloc(1, sizeof(iftListArray));

    la->size = arraySize;
    la->list = (iftList **) calloc(arraySize, sizeof(iftList *));
    for (int i = 0; i < arraySize; ++i)
        la->list[i] = iftCreateList();

    return la;
}

void iftBuildActiveRDS(iftUnlabeledSelector * selector, iftKnnGraph *graph){
    iftDataSet * Z = selector->Z;

    iftUSInitClusterData(selector, graph);

    // Find size of each cluster to build arrays
    int * clusterSizes = iftAllocIntArray(Z->nlabels);
    for (int s = 0; s < Z->nsamples; ++s)
        clusterSizes[Z->sample[s].label - 1]++;
    int ** clusterSamples = (int**)calloc(Z->nlabels, sizeof(int *));
    float ** clusterDist = (float**)calloc(Z->nlabels, sizeof(float *));
    for (int i = 0; i < Z->nlabels; ++i) {
        clusterSamples[i] = iftAllocIntArray(clusterSizes[i]);
        clusterDist[i] = iftAllocFloatArray(clusterSizes[i]);
    }

    // Fill in sample indexes and their distance to the root
    int * clusterIndex = iftAllocIntArray(Z->nlabels);
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
    for (int c = 0; c < Z->nlabels; ++c)
        iftFQuickSort(clusterDist[c], clusterSamples[c], 1, clusterSizes[c]-1, IFT_INCREASING);

    // Convert into list for easier use
    iftListArray *la = iftCreateListArray(Z->nlabels);
    for (int c = 0; c < Z->nlabels; ++c) {
        for (int s = 0; s < clusterSizes[c]; ++s)
            iftInsertListIntoTail(la->list[c], clusterSamples[c][s]);
    }
    selector->clusterList = la;

    // Clean up
    for (int c = 0; c < Z->nlabels; ++c) {
        free(clusterDist[c]);
        free(clusterSamples[c]);
    }
    free(clusterDist);
    free(clusterSamples);
    free(clusterSizes);
    free(clusterIndex);
}

iftActiveClassifier * iftCreateActiveClassifier(
        iftClassifierAlgorithm algo,
        iftDataSet * Z,
        iftUnlabeledSelector * semisupSelector)
{
    iftActiveClassifier * classifier = (iftActiveClassifier *)calloc(1, sizeof(*classifier));
    classifier->algo = algo;
    classifier->isTrained = false;
    classifier->OPFClassifier = NULL;
    classifier->LogRegClassifier = NULL;
    classifier->Z = Z;
    classifier->Zlabeled = NULL;
    classifier->Zunlabeled = NULL;
    classifier->semisupSelector = semisupSelector;
    return classifier;
}

iftList * iftPickFromUnlabeledSelector(iftUS * selector,
                                       int nSamples,
                                       iftActiveClassifier * classifier){

    // Sanity test
    if (selector == NULL)
        iftError("Invalid Unlabeled Selector.", "iftPickFromUnlabeledSelector");

    iftList * selectedSamples = NULL;

    if (selector->isFirstIter) {
        selector->isFirstIter = false;
        switch (selector->firstIterBehavior) {
        case FIRST_ITER_ROOTS:
            selectedSamples = iftPickRootsUS(selector);
            return selectedSamples;
        default:
            break;
        };
    }

    switch(selector->algo) {
    case ALL_SAMPLES:
        break;
    case FULL_RANDOM:
        selectedSamples = iftPickFullRandomUS(selector, nSamples);
        break;
    case ACTIVE_RWS:
        break;
    case ACTIVE_RDS:
        selectedSamples = iftPickActiveRDS(selector, nSamples, classifier);
        break;
    default:
        iftError("Invalid active learning algorithm", "iftPickFromUnlabeledSelector");
    };
    return selectedSamples;
}

iftList * iftPickRootsUS(iftUnlabeledSelector * selector)
{
    iftList * selectedSamples = iftCreateList();
    for (int c = 0; c < selector->nClusters; ++c)
        iftInsertListIntoTail(selectedSamples, selector->rootArray[c]);
    return selectedSamples;
}

iftList * iftPickActiveRDS(iftUnlabeledSelector * selector,
                           int nSamples, iftActiveClassifier * classifier)
{
    iftListArray * la = selector->clusterList;
    iftDataSet * Z = selector->Z;
    int * rootArray = selector->rootArray;
    iftList * selectedSamples = iftCreateList();

    // Init cluster iterator info
    int * backendSteps = iftAllocIntArray(selector->nClusters);
    iftNode** clusterIter = (iftNode**)calloc(selector->nClusters, sizeof(iftNode *));
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

int iftRemoveListNode(iftList *L, iftNode **NodeRef, bool isReverse)
{
    if (L == NULL || NodeRef == NULL)
        iftError("List L or node NodeRef is NULL.", "iftExtractListNode");

    iftNode *N = *NodeRef;
    if (N == NULL)
        iftError("Trying to extract NULL node.", "iftExtractListNode");

    // Update client pointer
    if (isReverse)
        *NodeRef = N->previous;
    else
        *NodeRef = N->next;

    // Remove intended node from List
    if (N->previous != NULL)
        N->previous->next = N->next;
    else
        L->head = N->next;

    if (N->next != NULL)
        N->next->previous = N->previous;
    else
        L->tail = N->previous;

    // Extract value
    int elem = N->elem;
    L->n--;

    free(N);
    return elem;
}



void iftActiveClassifierClassifySample(iftActiveClassifier * classifier, iftDataSet * Ztest, int sample)
{
    switch (classifier->algo) {
    case OPFSUP_CLASSIFIER:
    case OPFSEMI_CLASSIFIER:
        iftFindMinimumCostAndClassifySample(classifier->OPFClassifier, Ztest, sample);
        break;
    case LOGREG_CLASSIFIER:
        //iftLogRegSGDClassifySample(classifier->LogRegClassifier, Ztest, sample);
        break;
    default:
        break;
    };
}

void iftTrainActiveClassifier(iftActiveClassifier * classifier)
{
    // Select labeled and unlabeled samples
    iftResetActiveClassifierDataSets(classifier);
    classifier->Zlabeled = iftExtractSamples(classifier->Z, IFT_TRAIN);
    if (iftIsSemiSupClassifier(classifier))
        classifier->Zunlabeled = iftExtractUnlabeledSemisupSamples(classifier);

    // Train classifier
    iftResetActiveClassifierClassifiers(classifier);
    switch (classifier->algo) {
    case OPFSUP_CLASSIFIER:
        iftTrainActiveClassifierOPFSup(classifier);
        break;
    case OPFSEMI_CLASSIFIER:
        iftTrainActiveClassifierOPFSemi(classifier);
        break;
    case LOGREG_CLASSIFIER:
        //iftTrainActiveClassifierLogReg(classifier);
        break;
    default:
        break;
    };
    classifier->isTrained = true;
}

void iftResetActiveClassifierDataSets(iftActiveClassifier * classifier)
{
    iftDestroyDataSet(&(classifier->Zlabeled));
    iftDestroyDataSet(&(classifier->Zunlabeled));
}

bool iftIsSemiSupClassifier(iftActiveClassifier * classifier)
{
    return (classifier->algo == OPFSEMI_CLASSIFIER);
}

iftDataSet * iftExtractUnlabeledSemisupSamples(iftActiveClassifier * classifier)
{
    iftDataSet * Z = classifier->Z;
    iftDataSet * Zunlabeled = NULL;
    iftUnlabeledSelector* semisupSelector = classifier->semisupSelector;
    int nSelect = classifier->Zlabeled->nsamples * 2;

    iftList * unlabeledSelected =
            iftPickFromUnlabeledSelector(semisupSelector, nSelect, classifier);

    if (unlabeledSelected->n == 0) {
        // All samples are already labeled, convert to Supervised algorithm
        classifier->algo = OPFSUP_CLASSIFIER;
    } else {
        // Clean status label for extraction
        for (int s = 0; s < Z->nsamples; ++s)
            if (Z->sample[s].status != IFT_TRAIN)
                Z->sample[s].status = IFT_UNKNOWN;

        // Mark selected samples as IFT_TEST
        iftNode * listIter = unlabeledSelected->head;
        while (listIter != NULL) {
            int sample = iftRemoveListNode(unlabeledSelected, &listIter, false);
            Z->sample[sample].status = IFT_TEST;
        }

        Zunlabeled = iftExtractSamples(Z, IFT_TEST);
        iftSetStatus(Zunlabeled, IFT_TRAIN);
    }

    iftDestroyList(&unlabeledSelected);
    return Zunlabeled;
}

void iftResetActiveClassifierClassifiers(iftActiveClassifier * classifier)
{
    if (classifier->OPFClassifier != NULL) {
        // For convenience, the classifier graph always holds its own copy of Z
        iftDestroyDataSet(&(classifier->OPFClassifier->Z));
        iftDestroyCplGraph(&(classifier->OPFClassifier));
    }
    // TODO add this NULL test inside destroy function
    if (classifier->LogRegClassifier != NULL)
        iftDestroyLogReg(&(classifier->LogRegClassifier));
}

void iftTrainActiveClassifierOPFSup(iftActiveClassifier * classifier)
{
    iftDataSet * Ztrain = iftCopyDataSet(classifier->Zlabeled, true);
    classifier->OPFClassifier = iftCreateCplGraph(Ztrain);
    iftSupTrain(classifier->OPFClassifier);
}

void iftTrainActiveClassifierOPFSemi(iftActiveClassifier * classifier)
{
    // Avoid merge dataset problems

    classifier->Zunlabeled->nlabels = 0;
    classifier->OPFClassifier = iftSemiSupTrain(classifier->Zlabeled, classifier->Zunlabeled, FALSE);
}

int iftActiveClassifierClassifyDataSet(iftActiveClassifier * classifier, iftDataSet * Ztest)
{
    int n = 0;
    switch (classifier->algo) {
    case OPFSUP_CLASSIFIER:
    case OPFSEMI_CLASSIFIER:
        n = iftClassify(classifier->OPFClassifier, Ztest);
        break;
    case LOGREG_CLASSIFIER:
        //iftLogRegSGDClassify(classifier->LogRegClassifier, Ztest);
    default:
        break;
    };
    return n;
}

iftList * iftPickFullRandomUS(iftUnlabeledSelector * selector, int nSamples)
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

void iftResetRandomSelector(iftRandomSelector * rs)
{
    rs->currentSize = rs->totalSize;
}

int iftPickFromRandomSelector(iftRandomSelector * rs, const bool hasReposition)
{
    if (rs->currentSize <= 0)
        iftError("Random Selector pool is already empty", "iftPickFromRandomSelector");

    int i = iftRandomInteger(0, rs->currentSize - 1);
    int val = rs->nums[i];

    if(!hasReposition) {
        iftSwap(rs->nums[i], rs->nums[rs->currentSize - 1]);
        rs->currentSize -= 1;
    }

    return val;
}

