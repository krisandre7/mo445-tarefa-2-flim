#include "iftExperimentUtility.h"
#ifdef HAS_METIS
  #include <metis.h>
#endif

// ---------- External Methods to be moved ----------
/**
 * @brief Draws the borders of a single label.
 * @author Felipe Lemes Galvao
 * @date June 28, 2017
 *
 * @param[in,out] img Image to be drawn.
 * @param[in] labelMap Some label map associated with img.
 * @param[in] label Which label borders we want to draw.
 * @param[in] A Adjacency to define which pixels are borders.
 * @param[in] YCbCr Color value of drawn borders.
 * @param[in] brushShape Adjacency to choose which pixels around the
 *            ones defined as border pixels will be drawn.
 *
 * @pre All input values are non-NULL.
 * @pre The image and its label map have the same domain.
 * @pre The chosen label exists.
 */
void iftDrawBordersSingleLabel(iftImage *img, iftImage *labelMap,   int label, iftColor YCbCr)
{
  assert(img != NULL);
  assert(labelMap != NULL);
  assert(label > 0);
  assert(label <= iftMaximumValue(labelMap));
  assert(img->xsize == labelMap->xsize);
  assert(img->ysize == labelMap->ysize);
  assert(img->zsize == labelMap->zsize);

  int maxRangeValue = iftNormalizationValue(iftMaximumValue(img));
  iftAdjRel *A = iftCircular(1.0);
  iftAdjRel *brushShape = iftCircular(1.0);
  
  for (int p = 0; p < img->n; ++p) {
    if (labelMap->val[p] != label)
      continue;

    iftVoxel u = iftGetVoxelCoord(labelMap, p);

    for (int i = 1; i < A->n; ++i) {
      iftVoxel v = iftGetAdjacentVoxel(A, u, i);
      if (iftValidVoxel(labelMap, v)) {
        int q = iftGetVoxelIndex(labelMap, v);

        if (label != labelMap->val[q]) {
          iftDrawPoint(img, u, YCbCr, brushShape, maxRangeValue);
          break;
        }
      }
    }
  }
}

// ---------- Private Methods Declaration ----------
iftImage *iftMImageMedoidPixels(  iftMImage *mimg,   iftImage *spLabelMap);

// ---------- Private Methods Implementation ----------
iftImage *iftMImageMedoidPixels(  iftMImage *mimg,   iftImage *spLabelMap)
{
  assert(mimg != NULL);
  assert(spLabelMap != NULL);
  assert(iftIsDomainEqual(mimg, spLabelMap));

  // Calculate superpixel mean
  int nSuperpixels = iftMaximumValue(spLabelMap);
  iftMatrix *meanMx = iftCreateMatrix(mimg->m, nSuperpixels);
  int *superpixelSize = iftAllocIntArray(nSuperpixels);
  for (int p = 0; p < mimg->n; ++ p) {
    if (spLabelMap->val[p] == 0)
      continue;
    int superpixel = spLabelMap->val[p] - 1;
    superpixelSize[superpixel] += 1;
    for (int f = 0; f < mimg->m; ++f) {
      int index = iftGetMatrixIndex(meanMx, f, superpixel);
      meanMx->val[index] += mimg->val[p][f];
    }
  }
  for (int row = 0; row < meanMx->nrows; ++row) {
    for (int col = 0; col < meanMx->ncols; ++col) {
      int index = iftGetMatrixIndex(meanMx, col, row);
      meanMx->val[index] /= ((float) superpixelSize[row]);
    }
  }

  // Find pixel closest to mean in each superpixel
  int *selectedPixel = iftAllocIntArray(nSuperpixels);
  float *selectedPixelDistance = iftAllocFloatArray(nSuperpixels);
  for (int i = 0; i < nSuperpixels; ++i) {
    selectedPixel[i] = -1;
    selectedPixelDistance[i] = IFT_INFINITY_FLT;
  }
  for (int p = 0; p < mimg->n; ++p) {
    if (spLabelMap->val[p] == 0)
      continue;

    int superpixel = spLabelMap->val[p] - 1;
    // Calculate distance to mean (we ignore the sqrt)
    float distanceToMean = 0.0;
    for (int f = 0; f < mimg->m; ++f) {
      int index = iftGetMatrixIndex(meanMx, f, superpixel);
      double diff = (meanMx->val[index] - mimg->val[p][f]);
      distanceToMean += diff * diff;
    }
    // Check if new best is found
    if (distanceToMean < selectedPixelDistance[superpixel]) {
      selectedPixel[superpixel] = p;
      selectedPixelDistance[superpixel] = distanceToMean;
    }
  }

  // Build seed image from medoids
  iftImage *seeds = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
  int label = 1;
  for (int i = 0; i < nSuperpixels; ++i)
    seeds->val[selectedPixel[i]] = label++;

  iftDestroyMatrix(&meanMx);
  free(superpixelSize);
  free(selectedPixel);
  free(selectedPixelDistance);
  return seeds;
}

// ---------- Public Methods Implementation ----------

void iftPrintSuperpixelSegmentationMetrics(iftImage *segLabelMap, iftImage *gtLabelMap)
{
  assert(segLabelMap != NULL);
  assert(gtLabelMap != NULL);
  assert(iftIsDomainEqual(segLabelMap, gtLabelMap));

  iftImage *segBorders = iftBorderImage(segLabelMap, 0);
  iftImage *gtBorders = iftBorderImage(gtLabelMap, 0);

  // Cast   away to avoid compiler errors
  printf("BR: %f \n", iftBoundaryRecall((iftImage *) gtBorders,
        (iftImage *) segBorders, 2.0));
  printf("UE: %f \n", iftUnderSegmentation(gtLabelMap, segLabelMap));
  printf("Comp: %f \n", iftCompactness2D(segLabelMap));
  printf("Top: %f \n", iftTopologyMeasure(segLabelMap));

  iftDestroyImage(&segBorders);
  iftDestroyImage(&gtBorders);
}

iftImage *iftOverlaySegmentationBorders(iftImage *img, iftImage *segLabelMap, iftColor YCbCr)
{
  assert(img != NULL);
  assert(segLabelMap != NULL);
  assert(iftIsDomainEqual(img, segLabelMap));

  iftImage *res = iftCopyImage(img);
  iftAdjRel *A = iftCircular(1.0);
  iftAdjRel *B = iftCircular(0.0);
  iftDrawBorders(res, segLabelMap, A, YCbCr, B);

  iftDestroyAdjRel(&A);
  iftDestroyAdjRel(&B);
  return res;
}

iftMatrix *iftCountLabelsInSuperpixels(  iftImage *spLabelMap,   iftImage *baseLabelMap, bool isNormalized)
{
  assert(spLabelMap != NULL);
  assert(baseLabelMap != NULL);
  assert(iftIsDomainEqual(spLabelMap, baseLabelMap));

  // Extract implicit info from label maps
  int nSuperpixels = iftMaximumValue(spLabelMap);
  int nLabels = iftMaximumValue(baseLabelMap) + 1; // + background 0
  iftMatrix *labelCount = iftCreateMatrix(nLabels, nSuperpixels);

  // Count the labels in each superpixel
  for (int p = 0; p < spLabelMap->n; ++p) {
    // Skip pixel with no associated superpixel
    if (baseLabelMap->val[p] == 0)
      continue;

    int superpixel = spLabelMap->val[p] - 1;
    int label = baseLabelMap->val[p];
    int index = iftGetMatrixIndex(labelCount, label, superpixel);
    labelCount->val[index] += 1.0;
  }

  if (isNormalized) {
    for (int row = 0; row < labelCount->nrows; ++row) {
      float sum = 0.0;
      for (int col = 0; col < labelCount->ncols; ++col) {
        int index = iftGetMatrixIndex(labelCount, col, row);
        sum += labelCount->val[index];
      }
      for (int col = 0; col < labelCount->ncols; ++col) {
        int index = iftGetMatrixIndex(labelCount, col, row);
        labelCount->val[index] /= sum;
      }
    }
  }

  return labelCount;
}

iftImage *iftSuperpixelToMajoritySegmentation(  iftImage *segLabelMap,   iftImage *gtLabelMap)
{
  assert(segLabelMap != NULL);
  assert(gtLabelMap != NULL);
  assert(iftIsDomainEqual(segLabelMap, gtLabelMap));

  // Copy only to get domain, we will overwrite all values
  iftImage *res = iftCopyImage(segLabelMap);

  // Extract implicit info from label maps. 
  iftMatrix *labelCount = iftCountLabelsInSuperpixels(segLabelMap, gtLabelMap, false);

  // Find the mode of each superpixel
  int *superpixelMode = iftAllocIntArray(labelCount->nrows); 
  for (int i = 0; i < labelCount->nrows; ++i) {
    float *superpixelRow = iftMatrixRowPointer(labelCount, i);
    superpixelMode[i] = iftFArgmax(superpixelRow, labelCount->ncols);
  }

  // Map each superpixel to its mode
  for (int p = 0; p < res->n; ++p) {
    // Pixels without superpixels are assigned background by default
    if (segLabelMap->val[p] == 0) {
      res->val[p] = 0;
      continue;
    }

    int superpixel = segLabelMap->val[p] - 1;
    res->val[p] = superpixelMode[superpixel];
  }

  iftDestroyMatrix(&labelCount);
  free(superpixelMode);
  return res;
}

float iftAchievableSegmentationAccuracy(  iftImage *segLabelMap,   iftImage *gtLabelMap)
{
  assert(segLabelMap != NULL);
  assert(gtLabelMap != NULL);
  assert(iftIsDomainEqual(segLabelMap, gtLabelMap));

  iftImage *bestSeg = iftSuperpixelToMajoritySegmentation(segLabelMap, gtLabelMap);
  int correctPixels = 0;
  for (int p = 0; p < gtLabelMap->n; ++p)
    if (bestSeg->val[p] == gtLabelMap->val[p])
      correctPixels += 1;

  float ASA = (float) correctPixels / (float) gtLabelMap->n;
  iftDestroyImage(&bestSeg);
  
  return ASA;
}

float iftBoundaryPrecision(  iftImage *segBorders,   iftImage *gtBorders, float toleranceDist)
{
  assert(segBorders != NULL);
  assert(gtBorders != NULL);
  assert(iftIsDomainEqual(segBorders, gtBorders));
  assert(toleranceDist >= 0.0);

  iftAdjRel *A = NULL;
  if (iftIs3DImage(segBorders))
    A = iftSpheric(toleranceDist);
  else
    A = iftCircular(toleranceDist);

  int truePositives = 0;
  int nPositives = 0;
  for (int p = 0; p < segBorders->n; ++p) {
    if (segBorders->val[p] == 0)
      continue;

    nPositives += 1;

    iftVoxel u = iftGetVoxelCoord(segBorders, p);
    for (int i = 0; i < A->n; ++i) {
      iftVoxel v = iftGetAdjacentVoxel(A, u, i);
      if (!iftValidVoxel(segBorders, v))
        continue;

      int q = iftGetVoxelIndex(segBorders, v);
      if (gtBorders->val[q] != 0) {
        truePositives += 1;
        break;
      }
    }
  }

  assert(nPositives > 0.0);
  return (float)truePositives / (float)nPositives;
}

float iftBoundaryFScore(  iftImage *segBorders,   iftImage *gtBorders, float toleranceDist)
{
  assert(segBorders != NULL);
  assert(gtBorders != NULL);
  assert(iftIsDomainEqual(segBorders, gtBorders));
  assert(toleranceDist >= 0.0);

  float recall = iftBoundaryRecall((iftImage *) gtBorders,
      (iftImage *) segBorders, toleranceDist);
  float precision = iftBoundaryPrecision(segBorders, gtBorders, toleranceDist);

  if (recall * precision < IFT_EPSILON)
    return 0.0f;

  return (2.0f * precision * recall) / (precision + recall);
}

float iftDLogEncoding(float val, bool isNormalized)
{
  if (isNormalized)
    val *= 255.0;

  if (val <= 0.0)
    return 0.0;
  if (val <= 1.0)
    return 1.0;

  return (float)(ceil((log(val)/log(2))) + 1);
}

void iftSetDataSetGroupFromHierarchy(iftDataSet *Z,   iftImage *superpixelLabelMap,   iftImage *prevSegLabelMap)
{
  if (prevSegLabelMap != NULL) {
    assert(iftMaximumValue(prevSegLabelMap) == Z->nsamples);
    for (int p = 0; p < prevSegLabelMap->n; ++p) {
      int s = prevSegLabelMap->val[p] - 1;
      if (s < 0)
        continue;

      Z->sample[s].group = superpixelLabelMap->val[p];
    }
  } else {
    // Note: Assuming 1:1 correspondence of samples to pixels
    assert(Z->nsamples == superpixelLabelMap->n);
    for (int s = 0; s < Z->nsamples; ++s)
      Z->sample[s].group = superpixelLabelMap->val[s];
  }
}

// Fraction of pixels classified as class col in superpixel row
// Index 0 is used to store max value from the respective row/column
iftMatrix * iftSuperpixelClassificationResults(iftDataSet *Z,   iftImage *superpixelLabelMap,   iftImage *prevSegLabelMap)
{
  int nSuperpixels = iftMaximumValue(superpixelLabelMap);
  int *spSize = iftAllocIntArray(nSuperpixels+1);
  iftMatrix *mx = iftCreateMatrix(Z->nclasses+1, nSuperpixels+1);

  iftSetDataSetGroupFromHierarchy(Z, superpixelLabelMap, prevSegLabelMap);

  for (int s = 0; s < Z->nsamples; ++s) {
    int spLabel = Z->sample[s].group;
    iftMatrixElem(mx, Z->sample[s].label, spLabel) += 1.0;
    spSize[spLabel] += 1;
  }

  for (int row = 0; row < mx->nrows; ++row) {
    // Empty superpixel label
    if (spSize[row] <= 0)
      continue;

    for (int col = 0; col < mx->ncols; ++col)
      iftMatrixElem(mx, col, row) /= (float) spSize[row];
  }

  // Use extra row/col to max %
  for (int row = 1; row < mx->nrows; ++row) {
    for (int col = 1; col < mx->ncols; ++col) {
      if (iftMatrixElem(mx, col, row) > iftMatrixElem(mx, 0, row))
        iftMatrixElem(mx, 0, row) = iftMatrixElem(mx, col, row);
      if (iftMatrixElem(mx, col, row) > iftMatrixElem(mx, col, 0))
        iftMatrixElem(mx, col, 0) = iftMatrixElem(mx, col, row);
    }
  }

  // Clean up
  free(spSize);

  return mx;
}

// First element has weight n, second n-1 etc
int iftRandomSelectionWeightedByOrder(int n)
{
  // Avoid overflow
  assert(n < (int) sqrt(INT_MAX));

  int sum = (n * (n + 1)) / 2;
  int val = rand() % sum;
  int s = 0;

  while (1) {
    if (val < n - s)
      return s;
    else
      val -= (n - s++);
  }

  assert(false);
  return -1;
}

iftSet *iftFindRepresentativePixels(  iftImage *gtLabelMap)
{
  assert(gtLabelMap != NULL);

  int nLabels = iftMaximumValue(gtLabelMap) + 1; // Includes bg = 0
  int *pxChosen = iftAllocIntArray(nLabels);
  int *pxDist = iftAllocIntArray(nLabels);
  for (int i = 0; i < nLabels; ++i)
    pxDist[i] = IFT_INFINITY_INT;


  iftAdjRel *A = iftIs3DImage(gtLabelMap) ? iftSpheric(1.0) : iftCircular(1.0);
  iftImage *borderDistSqr = iftBorderDistTrans((iftImage *) gtLabelMap, A);

  for (int p = 0; p < nLabels; ++p) {
    int label = gtLabelMap->val[p];
    if (borderDistSqr->val[p] >= pxDist[label])
      continue;
    pxChosen[label] = p;
    pxDist[label] = borderDistSqr->val[p];
  }

  iftSet *res;
  for (int i = 0; i < nLabels; ++i)
    iftInsertSet(&res, pxChosen[i]);

  return res;
}

float iftSpIGraphNodeToSeedFeatDist(iftSuperpixelIGraph *igraph, int node, int seed)
{
  return igraph->distFun(
      iftMatrixRowPointer(igraph->feats, node),
      iftMatrixRowPointer(igraph->seedFeats, seed),
      igraph->distAlpha,
      igraph->feats->ncols);
}

float iftSpIGraphNodeToSeedSpaceDist(iftSuperpixelIGraph *igraph, int node, int seed)
{
  return iftFeatDistance(
      iftMatrixRowPointer(igraph->pos, node),
      iftMatrixRowPointer(igraph->seedPos, seed),
      igraph->pos->ncols);
}

iftImage *iftForceLabelMapConnectivity(iftImage *labelMap, int minSize)
{
  // Init
  iftImage *res = iftCreateImage(labelMap->xsize, labelMap->ysize, labelMap->zsize);
  iftAdjRel *A = iftIs3DImage(labelMap) ? iftSpheric(1.0f) : iftCircular(1.0f);
  iftFIFO *Q = iftCreateFIFO(res->n);

  // Each connected component is given a unique label if their size > minSize
  int label = 1;
  for (int pTop = 0; pTop < res->n; ++pTop) {
    // Connected component already processed
    if (res->val[pTop] > 0)
      continue;

    // Initialization
    iftResetFIFO(Q);
    iftInsertFIFO(Q, pTop);
    int size = 0;
    int neighLabel = 0;
    iftSet *componentVoxels = NULL;

    // Cover entire component with BFS
    res->val[pTop] = label;
    while (!iftEmptyFIFO(Q)) {
      // Process voxel
      int p = iftRemoveFIFO(Q);
      iftInsertSet(&componentVoxels, p);
      size += 1;

      // Visit neighbors
      iftVoxel u = iftGetVoxelCoord(res, p);
      for (int i = 1; i < A->n; ++i) {
        iftVoxel v = iftGetAdjacentVoxel(A, u, i);
        if (!iftValidVoxel(res, v))
          continue;
        int q = iftGetVoxelIndex(res, v);
        if (res->val[q] == 0) {
          if (labelMap->val[p] == labelMap->val[q]) {
            res->val[q] = label;
            iftInsertFIFO(Q, q);
          }
        } else if (res->val[q] != label) {
          neighLabel = res->val[q];
        }
      }
    }

    // Relabel with neighbor if too small (except first one)
    if (size < minSize && neighLabel > 0) {
      while (componentVoxels != NULL) {
        int p = iftRemoveSet(&componentVoxels);
        res->val[p] = neighLabel;
      }
    } else {
      label += 1;
      iftDestroySet(&componentVoxels);
    }
  }

  iftDestroyAdjRel(&A);
  iftDestroyFIFO(&Q);

  return res;
}

iftImage * iftBuildMetisInitPartition(iftSuperpixelIGraph *igraph, int nSp)
{
#ifdef HAS_METIS
  if (igraph->type != EXPLICIT)
    iftError("iftBuildMetisInitPartition", "Only implemented for explicit graphs.");

  // Get target num of regions
  idx_t mOptions[METIS_NOPTIONS];
  METIS_SetDefaultOptions(mOptions);
  mOptions[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
  mOptions[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
  mOptions[METIS_OPTION_CONTIG] = 1; // Force connected regions
  mOptions[METIS_OPTION_COMPRESS] = 0; // Do not compress graph
  mOptions[METIS_OPTION_NUMBERING] = 0; // C-style indexes

  // General allocation
  idx_t nvtxs = igraph->nNodes;
  idx_t ncon = 1;
  idx_t *xadj = (idx_t *) malloc((igraph->nNodes + 1) * sizeof(*xadj));
  int nEdges = 0;
  for (int i = 0; i < igraph->nNodes; ++i) {
    xadj[i] = nEdges;
    nEdges += iftSetSize(igraph->nodeAdj[i]);
  }
  xadj[igraph->nNodes] = nEdges;
  idx_t *adjncy = (idx_t *) malloc(nEdges * sizeof(*adjncy));
  idx_t *vwgt = (idx_t *) malloc(igraph->nNodes * sizeof(*vwgt));
  idx_t *adjwgt = (idx_t *) malloc(nEdges * sizeof(*adjwgt));
  idx_t nparts = nSp;
  idx_t objval;
  idx_t *part = (idx_t *) malloc(igraph->nNodes * sizeof(*part));

  // Init adjacency and weights
  int idx = 0;
  float *fltAdjwgt = iftAllocFloatArray(nEdges);
  float maxWeight = 0.0f;
  for (int i = 0; i < igraph->nNodes; ++i) {
    vwgt[i] = igraph->spSize[i];
    //vwgt[i] = 1;
    for (iftSet *s = igraph->nodeAdj[i]; s != NULL; s = s->next) {
      adjncy[idx] = s->elem;
      fltAdjwgt[idx] = igraph->distFun(
          iftMatrixRowPointer(igraph->feats, i),
          iftMatrixRowPointer(igraph->feats, s->elem),
          igraph->distAlpha,
          igraph->feats->ncols);
      if (maxWeight < fltAdjwgt[idx])
        maxWeight = fltAdjwgt[idx];
      idx++;
    }
  }
  // Discretize edge weights
  int RANGE = 1 << 24;
  for (int e = 0; e < nEdges; ++e) {
    adjwgt[e] = (int) round((fltAdjwgt[e] / maxWeight) * RANGE);
    if (adjwgt[e] <= 0)
      adjwgt[e] = 1;
  }
  
  int status = METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, vwgt, NULL, adjwgt, &nparts, NULL, NULL, mOptions, &objval, part); 
  switch(status) {
    case METIS_OK:
      break;
    case METIS_ERROR_INPUT:
      iftError("METIS_PartGraphKway", "Input error.");
    case METIS_ERROR_MEMORY:
      iftError("METIS_PartGraphKway", "Memory error.");
    case METIS_ERROR:
      iftError("METIS_PartGraphKway", "General error.");
    default:
      iftError("METIS_PartGraphKway", "Undefined error.");
      break;
  }

  // Convert to label map
  iftImage *res = iftCopyImage(igraph->refImg);
  for (int p = 0; p < res->n; ++p) {
    int node = igraph->refImg->val[p] - 1;
    if (node < 0)
      continue;
    res->val[p] = part[node] + 1;
  }
  iftWriteImageP2(res, "metis_init.pgm");

  return res;
#else
  iftError("iftBuildMetisInitPartition", "Demo compiled without METIS.");
  return NULL;
#endif
}

extern bool _iftHasCSVHeader(const char *csv_pathname, char separator);
extern void _iftCountNumOfRowsAndColsFromCSVFile(const char *csv_pathname, long *nrows, long *ncols, char separator);
extern iftCSV *_iftCreateCSVWithoutStringAllocation(long nrows, long ncols);
iftCSV *iftReadCSVWithHeader(const char *csv_pathname, const char separator, bool *has_header) 
{
    if (!iftFileExists(csv_pathname))
        iftError("The CSV file pathname \"%s\" does not exist!", "iftReadCSV", csv_pathname);
    
    char strSeparator[2] = {separator, '\0'};
    
    *has_header = _iftHasCSVHeader(csv_pathname, separator);
    
    long nrows, ncols;
    _iftCountNumOfRowsAndColsFromCSVFile(csv_pathname, &nrows, &ncols, separator);
    
    iftCSV *csv = _iftCreateCSVWithoutStringAllocation(nrows, ncols);
    
    FILE *fp = fopen(csv_pathname, "rb");
    if (fp == NULL)
        iftError(MSG_FILE_OPEN_ERROR, "_iftCountNumOfRowsAndColsFromCSVFile", csv_pathname);
    
    // copies the values from the CSV file
    iftSList *SL = NULL;
    char *line = iftGetLine(fp);
    
    long i = 0;
    while (line != NULL) {
        SL = iftSplitString(line, strSeparator);
        
        for (long j = 0; j < csv->ncols; j++) {
            csv->data[i][j] = iftRemoveSListHead(SL); // just points to string
            // removes the '\n' and '\r' from the paths
            iftRightTrim(csv->data[i][j], '\n');
            iftRightTrim(csv->data[i][j], '\r');
        }
        i++;
        iftDestroySList(&SL);
        
        iftFree(line);
        line = iftGetLine(fp);
    }
    
    fclose(fp);
    
    return csv;
}

int *iftGetSuperpixelSizes(  iftImage *labelMap, int nSp)
{
  if (nSp <= 0)
    nSp = iftMaximumValue(labelMap);
  int *sizes = iftAllocIntArray(nSp);
  
  for (int p = 0; p < labelMap->n; ++p) {
    int label = labelMap->val[p] - 1;
    if (label >= 0)
      sizes[label] += 1;
  }

  return sizes;
}

float *iftGetSuperpixelObjectOverlap(  iftImage *labelMap,   iftImage *gt, int nSp)
{
  if (nSp <= 0)
    nSp = iftMaximumValue(labelMap);
  int *sizes = iftGetSuperpixelSizes(labelMap, nSp);
  int *objCount = iftAllocIntArray(nSp);
  float *overlap = iftAllocFloatArray(nSp);
  int minGtLabel = iftMinimumValue(gt);

  for (int p = 0; p < labelMap->n; ++p) {
    if (gt->val[p] <= minGtLabel)
      continue;

    int label = labelMap->val[p] - 1;
    if (label < 0)
      continue;

    objCount[label] += 1;
  }

  for (int i = 0; i < nSp; ++i)
    overlap[i] = ((float) objCount[i]) / ((float) sizes[i]);

  free(sizes);
  free(objCount);

  return overlap;
}

