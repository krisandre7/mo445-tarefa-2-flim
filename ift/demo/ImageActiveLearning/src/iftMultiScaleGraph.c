#include "iftMultiScaleGraph.h"

// Private function in iftRISF.c
iftMatrix * iftComputeSuperpixelFeaturesByColorSpaceMean(  iftImage *superpixelLabelMap,   iftImage *img, iftColorSpace colorSpace);

// ---------- Private Methods Declaration ----------
/*
 * @brief Adds edges across each pair of scales.
 * @author Felipe Lemes Galvao
 * @date September, 2018
 */
void iftCreateMixedMultiScaleAdj(iftMultiScaleSpGraph *graph);

/*
 * @brief Encodes scale and superpixel in a single int.
 * @author Felipe Lemes Galvao
 * @date October, 2018
 */
int iftEncodeScaleAndSp(int scale, int sp);

/*
 * @author Felipe Lemes Galvao
 * @date October, 2018
 */
void iftFindPixelActiveScaleAndSp(iftMultiScaleSpGraph *graph, int p, int *outScale, int *outSp);

/*
 * @author Felipe Lemes Galvao
 * @date October, 2018
 */
void iftClearMultiScaleSpGraphLabels(iftMultiScaleSpGraph *graph, int val);

// ---------- Private Methods Implementation ----------
void iftCreateMixedMultiScaleAdj(iftMultiScaleSpGraph *graph)
{
  assert(graph != NULL);
  assert(graph->refImgInScale != NULL);
  assert(graph->nScales > 0);
  for (int i = 0; i < graph->nScales; ++i)
    assert(graph->refImgInScale[i] != NULL);
  
  // Shorthand and main ref
  iftImage *img = graph->refImgInScale[0];

  iftAdjRel *A = iftIs3DImage(img) ? iftSpheric(1.0) : iftCircular(1.0);

  for (int p = 0; p < img->n; ++p) {
    if (img->val[p] == 0)
      continue;
    
    iftVoxel u = iftGetVoxelCoord(img, p);
    for (int k = 0; k < A->n; ++k) {
      iftVoxel v = iftGetAdjacentVoxel(A, u, k);
      if (!iftValidVoxel(img, v))
        continue;
      int q = iftGetVoxelIndex(img, v);
      if (img->val[q] == 0)
        continue;

      for (int i = 0; i < graph->nScales; ++i) {
        iftImage *refImg = graph->refImgInScale[i];
        if (refImg->val[p] == refImg->val[q])
          continue;

        int baseSp = refImg->val[p] - 1;
        int baseNode = iftEncodeScaleAndSp(baseSp, i);

        for (int j = i; j < graph->nScales; ++j) {
          /* Adds edge in both directions so processing can go
           *  strictly from coarser to finer segmentation. */
          /* We assume that the number of neighbors << number of nodes
           *  so that traversing the entire node adjacency each time
           *  is not a problem. */
          int targetSp = graph->refImgInScale[j]->val[q] - 1;
          int targetNode = iftEncodeScaleAndSp(targetSp, j);
          iftUnionSetElem(&(graph->nodeAdjInScale[i][baseSp]), targetNode);
          iftUnionSetElem(&(graph->nodeAdjInScale[j][targetSp]), baseNode);
        }
      }
    }
  }
}

int iftEncodeScaleAndSp(int scale, int sp)
{
  return ((sp << 4) + scale);
}

void iftFindPixelActiveScaleAndSp(iftMultiScaleSpGraph *graph, int p, int *outScale, int *outSp)
{
  for (int i = 0; i < graph->nScales; ++i) {
    int sp = graph->refImgInScale[i]->val[p] - 1;
    if (!graph->activeNodeFlagInScale[i][sp])
      continue;

    *outScale = i;
    *outSp = sp;
    return;
  }
}

void iftClearMultiScaleSpGraphLabels(iftMultiScaleSpGraph *graph, int val)
{
  for (int i = 0; i < graph->nScales; ++i)
    for (int j = 0; j < graph->nNodesInScale[i]; ++j)
      graph->labelInScale[i][j] = val;
}

// ---------- Public Methods Implementation ----------
iftMultiScaleSpGraph *iftCreateMultiScaleGraph(int nScales, iftImage **spLabelImgInScale, iftImage *img)
{
  assert(nScales > 0);
  assert(spLabelImgInScale != NULL);
  for (int i = 0; i < nScales; ++i)
    assert(spLabelImgInScale[i] != NULL);
  assert(img != NULL);

  iftMultiScaleSpGraph *graph = malloc(sizeof(*graph));
  
  graph->nScales = nScales;

  graph->nNodesInScale = iftAllocIntArray(nScales);
  for (int i = 0; i < nScales; ++i) {
    graph->nNodesInScale[i] = iftMaximumValue(spLabelImgInScale[i]);
    assert(graph->nNodesInScale[i] > 0);
  }

  graph->activeNodeFlagInScale = 
    malloc(nScales * sizeof(*(graph->activeNodeFlagInScale)));
  for (int i = 0; i < nScales; ++i)
    graph->activeNodeFlagInScale[i] = iftAllocIntArray(graph->nNodesInScale[i]);
  /* Only coarsest scale nodes start active. */
  for (int s = 0; s < graph->nNodesInScale[0]; ++s)
    graph->activeNodeFlagInScale[0][s] = 1;

  /* Actual edges added in the end. */
  graph->nodeAdjInScale = 
    malloc(nScales * sizeof(*(graph->nodeAdjInScale)));
  for (int i = 0; i < nScales; ++i) {
    graph->nodeAdjInScale[i] = iftAlloc(graph->nNodesInScale[i],
        sizeof(*(graph->nodeAdjInScale[i])));
  }

  graph->featsInScale = 
    malloc(nScales * sizeof(*(graph->featsInScale)));
  for (int i = 0; i < nScales; ++i) {
    iftMatrix *feats = NULL; 
    if (iftIsColorImage(img)) {
      feats = iftComputeSuperpixelFeaturesByColorSpaceMean(
          spLabelImgInScale[i], img, LABNorm_CSPACE);  
    } else {
      feats = iftComputeSuperpixelFeaturesByColorSpaceMean(
          spLabelImgInScale[i], img, GRAYNorm_CSPACE);  
    }
    graph->featsInScale[i] = feats;
  }

  graph->refImgInScale =
    malloc(nScales * sizeof(*(graph->refImgInScale)));
  for (int i = 0; i < nScales; ++i)
    graph->refImgInScale[i] = iftCopyImage(spLabelImgInScale[i]);

  graph->labelInScale =
    malloc(nScales * sizeof(*(graph->labelInScale)));
  for (int i = 0; i < nScales; ++i)
    graph->labelInScale[i] = iftAllocIntArray(graph->nNodesInScale[i]);

  graph->markers = iftCreateImage(img->xsize, img->ysize, img->zsize);

  iftCreateMixedMultiScaleAdj(graph);

  return graph;
}

void iftExpandMultiScaleGraphSp(iftMultiScaleSpGraph *graph, int scale, int sp)
{
  assert(graph != NULL);
  assert(scale >= 0 && scale < graph->nScales);
  assert(sp >= 0 && sp < graph->nNodesInScale[scale]);

  if (scale == graph->nScales - 1) {
    iftWarning("Trying to expand superpixel from finest scale.", "iftExpandMultiScaleGraphSp");
    return;
  }

  // Exchange active flag from father to children
  graph->activeNodeFlagInScale[scale][sp] = 0;
  /* NOTE: If traversing the entire image each time is too expensive,
   *  it is possible to pre-compute the 'vertical' edges in the
   *  original hierarchy beforehand. */
  for (int p = 0; p < graph->refImgInScale[scale]->n; ++p) {
    if (graph->refImgInScale[scale]->val[p] - 1 != sp)
      continue;
    int finerSp = graph->refImgInScale[scale+1]->val[p] - 1;
    graph->activeNodeFlagInScale[scale+1][finerSp] = 1;
  }
}

void iftUpdateMultiScaleSpGraphByMarkers(iftMultiScaleSpGraph *graph)
{
  iftClearMultiScaleSpGraphLabels(graph, -1);

  int hasLabelConflict;
  do {
    hasLabelConflict = 0;
    for (int p = 0; p < graph->markers->n; ++p) {
      if (graph->markers->val[p] == 0)
        continue;

      int markerLabel = graph->markers->val[p] - 1;

      int scale = -1;
      int sp = -1;
      iftFindPixelActiveScaleAndSp(graph, p, &scale, &sp);

      if (graph->labelInScale[scale][sp] < 0)
        graph->labelInScale[scale][sp] = markerLabel;

      if (graph->labelInScale[scale][sp] != markerLabel) {
        hasLabelConflict = 1;
        iftExpandMultiScaleGraphSp(graph, scale, sp);
        break;
      }
    }
  } while (!hasLabelConflict);
}

iftImage *iftMultiScaleSpGraphFinerLabelMap(iftMultiScaleSpGraph *graph)
{
  iftClearMultiScaleSpGraphLabels(graph, 0);

  // Mark each target element with an unique label value
  int currentLabel = 0;
  for (int i = 1; i < graph->nScales+1; ++i) {
    for (int p = 0; p < graph->markers->n; ++p) {
      int parentSp = graph->refImgInScale[i-1]->val[p] - 1;
      if (!graph->activeNodeFlagInScale[i-1][parentSp])
        continue;

      int scale = (i < graph->nScales) ? i : i-1;
      int sp = graph->refImgInScale[scale]->val[p] - 1;
      if (graph->labelInScale[scale][sp] <= 0)
        graph->labelInScale[scale][sp] = ++currentLabel;
    }
  }

  // Build label image
  iftImage *aux = graph->markers; // Shorthand
  iftImage *res = iftCreateImage(aux->xsize, aux->ysize, aux->zsize);
  for (int i = 0; i < graph->nScales; ++i) {
    for (int p = 0; p < graph->markers->n; ++p) {
      int sp = graph->refImgInScale[i]->val[p] - 1;
      int label = graph->labelInScale[i][sp];
      if (!label)
        continue;

      res->val[p] = label;
    }
  }

  return res;
}

/* TODO check iftSuperpixelActiveLearning.c for 2-level consistency.
 * TODO re-appropriate RISF segmentation IFT to single shot seg
 * TODO clear unnecessary data (adj & feats)
 */
