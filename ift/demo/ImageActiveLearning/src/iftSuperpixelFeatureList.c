#include "iftSuperpixelFeatureList.h"

/** ---------- Public implementations ---------- */

void iftAddToSuperpixelFeatureList(iftSuperpixelFeatureList **listRef,   iftMatrix *superpixelFeatures, iftArcWeightFun distFun, const char *identifier,   int featId,   iftPathCostNodes pathCostNodes)
{
  iftSuperpixelFeatureList *newNode = iftAlloc(1, sizeof(*newNode));
  
  newNode->feats = iftCopyMatrix(superpixelFeatures);
  newNode->distFun = distFun;
  newNode->distAlpha = iftAllocFloatArray(superpixelFeatures->ncols);
  for (int i = 0; i < superpixelFeatures->ncols; ++i)
    newNode->distAlpha[i] = 1.0;
  newNode->alpha = 1.0;
  newNode->beta = 1.0;
  if (identifier)
    strncpy(newNode->id, identifier, sizeof(newNode->id));
  newNode->featId = featId;
  newNode->pathCostNodes = pathCostNodes;
  newNode->auxFeats = NULL;
  newNode->next = *listRef;
  *listRef = newNode;
}

void iftDestroySuperpixelFeatureList(iftSuperpixelFeatureList **listRef)
{
  if (listRef == NULL || *listRef == NULL)
    return;

  iftSuperpixelFeatureList *listNode = *listRef;
  while (listNode != NULL) {
    // Clear internal node data
    iftDestroyMatrix(&(listNode->feats));
    free(listNode->distAlpha);
    iftDestroyMatrix(&(listNode->auxFeats));

    // Free node and move to next one
    iftSuperpixelFeatureList *next = listNode->next;
    free(listNode);
    listNode = next;
  }

  *listRef = NULL;
}

void iftChangeSuperpixelFeatureWeight(iftSuperpixelFeatureList *listNode, float alpha, float beta)
{
  listNode->alpha = alpha;
  listNode->beta = beta;
}

void iftUpdateGraphBasedSuperpixelAuxFeats(iftSuperpixelFeatureList *listNode, iftForestAnnotationInfo *ann)
{
  if (listNode->distFun == NULL)
    return;

  switch(listNode->pathCostNodes) {
    // TODO Move to isolated function
    case IFT_MEAN_PATHCOST_NODES:
      // Check if new matrix has to be allocated
      if (listNode->auxFeats == NULL || 
          listNode->auxFeats->nrows != ann->nLabels ||
          listNode->auxFeats->ncols != listNode->feats->ncols)
      {
        iftDestroyMatrix(&(listNode->auxFeats));
        listNode->auxFeats =
          iftCreateMatrix(ann->nLabels, listNode->feats->ncols);
      } else {
        // Otherwise clear old values
        for (int i = 0; i < listNode->auxFeats->n; ++i)
          listNode->auxFeats->val[i] = 0.0;
      }

      iftMatrix *auxMx = listNode->auxFeats; // shorthand
      iftMatrix *featsMx = listNode->feats; // shorthand
      // Counter for how many nodes each label has
      int *labelCount = iftAllocIntArray(ann->nLabels);

      // Add each node feature to the label indexed feat matrix
      for (int r = 0; r < listNode->feats->nrows; ++r) {
        int label = ann->label[r] - 1;
        for (int c = 0; c < listNode->feats->ncols; ++c) {
          auxMx->val[iftGetMatrixIndex(auxMx, c, label)]
            += featsMx->val[iftGetMatrixIndex(featsMx, c, r)];
        }
        labelCount[label]++;
      }

      // Divide by node count to get actual mean
      for (int r = 0; r < auxMx->nrows; ++r)
        for (int c = 0; c < auxMx->ncols; ++c)
          auxMx->val[iftGetMatrixIndex(auxMx, c, r)] /= labelCount[r];

      break;
    default:
      // Most cases do not require additional features
      break;
  }
}
