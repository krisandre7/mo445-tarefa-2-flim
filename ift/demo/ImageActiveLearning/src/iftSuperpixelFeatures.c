#include "iftSuperpixelFeatures.h"

// ---------- Private Methods Declaration ----------
// Lapack function we use to solve the generalized symmetric definite
//  eigenproblem for the covariance distance
extern int ssygvd_(int *itype, char *jobz, char *uplo, int *n, float *a, int *lda, float *b, int *ldb, float *w, float *work, int *lwork, int *iwork, int *liwork, int *info);

float iftCovarianceDistance(float *f1, float *f2, float *alpha, int n)
{
  // Find order = ncols = nrows = sqrt(n)
  int order = (int) round(sqrt(n));
  assert(order * order == n);

  // We need a copy as the matrices values will be destroyed
  // No need to switch to column major representation (symmetric)
  float *A = iftAllocFloatArray(n);
  float *B = iftAllocFloatArray(n);
  for (int i = 0; i < n; ++i) {
    A[i] = f1[i];
    B[i] = f2[i];
  }

  // Param to choose the form A*x = (lambda)*B*x
  int type = 1;

  // Eigenvalues, the values we want to compute the distance
  float *lambda = iftAllocFloatArray(n);

  // Work arrays (minimum values)
  int lwork = 2*order + 1;
  float *work = iftAllocFloatArray(lwork);
  int liwork = 1;
  int *iwork = iftAllocIntArray(liwork); 

  // Execution feedback
  int info = -1;

  ssygvd_(
      &type, // Eigenvalue problem choice (A*x = lambda*B*x)
      "N", // Compute eigenvalues only
      "U", // Store upper triangle of the symmetric matrices
      &order, // Order of matrices
      A, // Input n x n matrix A
      &order, // A->nrows
      B, // Input n x n matrix B
      &order, // B->nrows
      lambda, // Output array of eigenvalues in ascending order
      work,
      &lwork,
      iwork,
      &liwork,
      &info); // Return code, 0 means sucess

  for (int i = 0; i < order; ++i)
    printf("Lambda[%d] = %f\n", i, lambda[i]);

  // Check for errors
  if (info != 0) {
    fprintf(stderr, "ssygvd_ returned error code %d.\n", info);
    exit(1);
  }

  // Compute actual distance
  double dist = 0.0;
  for (int i = 0; i < order; ++i) {
    assert(lambda[i] >= 0.0);
    double ln = log(lambda[i]);
    dist += (ln * ln);
  }
  dist = sqrt(dist);

  // Clean up
  free(A);
  free(B);
  free(lambda);
  free(work);
  free(iwork);

  return dist;
}

/**
 * @brief Computes histogram assignment of each pixel.
 * @author Felipe Lemes Galvao
 * @date January 5, 2017
 */
iftImage * iftColorHistogramAssignPixelBin(  iftImage *img, int binsPerColor);

/**
 * @brief Computes Interior and border mapping for BIC.
 * @author Felipe Lemes Galvao
 * @date January 5, 2017
 *
 * @param[in] binImage Obtained in iftColorHistogramAssignPixelBin.
 * @return Image with interior/border pixels with brightness 0/255.
 */
iftImage * iftBICMask(  iftImage *binImage);

/**
 * @brief Concatenates "horizontaly" two matrixes with same number of rows.
 * @author Felipe Lemes Galvao
 * @date January 5, 2017
 *
 * @details Can be made more general and included in iftMatrix.h/c.
 */
iftMatrix *iftConcatenateSuperpixelFeatMatrix(iftMatrix *A, iftMatrix *B);

/**
 * @brief Normalizes each row of a matrix independently.
 * @author Felipe Lemes Galvao
 * @date January 5, 2017
 *
 * @details Can be made more general and included in iftMatrix.h/c.
 */
void iftNormalizeSuperpixelFeatMatrix(iftMatrix *feats);

// ---------- Private Methods Implementation ----------
iftImage * iftColorHistogramAssignPixelBin(  iftImage *img, int binsPerColor)
{
  iftMImage *mimg = NULL;
  if (iftIsColorImage(img))
    mimg = iftImageToMImage(img, LABNorm_CSPACE);
  else
    mimg = iftImageToMImage(img, GRAYNorm_CSPACE);
  for (int b = 0; b < mimg->m; ++b) {
    float minB = mimg->val[0][b];
    for (int p = 0; p < mimg->n; ++p) {
      float val = mimg->val[p][b];
      if (val < minB)
        minB = val;
    }
    for (int p = 0; p < mimg->n; ++p) {
      mimg->val[p][b] += minB;
      if (mimg->val[p][b] < 0.0f)
        mimg->val[p][b] = 0.0f;
    }
    float max=IFT_INFINITY_FLT_NEG;
    for (int p=0; p < mimg->n; p++)
      if (mimg->val[p][b]>max)
	max = mimg->val[p][b];
    if (max > IFT_EPSILON)
      for (int p=0; p < mimg->n; p++)
	mimg->val[p][b] /= max;
  }

  iftImage *res = iftCreateImage(img->xsize, img->ysize, img->zsize);
  for (int p = 0; p < res->n; ++p) {
    int bin = 0;
    int mult = 1;
    for (int b = 0; b < mimg->m; ++b) {
      int dimBin = floor(mimg->val[p][b] * binsPerColor);
      /*if (dimBin < 0) {
        printf("dimBin = floor (%f * %d)\n", mimg->val[p][b], binsPerColor);
      }*/
      if (dimBin >= binsPerColor) // fix when val = 1.0
        dimBin = binsPerColor - 1;
      assert(dimBin >= 0);
      bin += dimBin * mult;
      mult *= binsPerColor;
    }
    res->val[p] = bin;
  }

  iftDestroyMImage(&mimg);
  return res;
}

iftImage * iftBICMask(  iftImage *binImage)
{
  iftImage *mask = iftCreateImage(binImage->xsize, binImage->ysize, binImage->zsize);
  iftAdjRel *A = iftCircular(1.0);
  for (int p = 0; p < mask->n; ++p) {
    iftVoxel u = iftGetVoxelCoord(mask, p);
    for (int i = 1; i < A->n; ++i) {
      iftVoxel v = iftGetAdjacentVoxel(A, u, i);
      int q = iftGetVoxelIndex(mask, v);

      if (!iftValidVoxel(mask, v) || binImage->val[p] != binImage->val[q]) {
        mask->val[p] = 255;
        break;
      }
    }
  }

  return mask;
}

iftMatrix *iftConcatenateSuperpixelFeatMatrix(iftMatrix *A, iftMatrix *B)
{
  assert(A->nrows == B->nrows);
  iftMatrix *res = iftCreateMatrix(A->ncols + B->ncols, A->nrows);
  for (int row = 0; row < A->nrows; ++row)
    for (int col = 0; col < A->ncols; ++col)
      res->val[iftGetMatrixIndex(res, col, row)] =
        A->val[iftGetMatrixIndex(A, col, row)];

  for (int row = 0; row < B->nrows; ++row)
    for (int col = 0; col < B->ncols; ++col)
      res->val[iftGetMatrixIndex(res, col + A->ncols, row)] =
        B->val[iftGetMatrixIndex(B, col, row)];

  return res;
}

void iftNormalizeSuperpixelFeatMatrix(iftMatrix *feats)
{
  for (int row = 0; row < feats->nrows; ++row) {
    double sum = 0.0;
    for (int col = 0; col < feats->ncols; ++col)
      sum += feats->val[iftGetMatrixIndex(feats, col, row)];
    assert(sum > IFT_EPSILON);
    for (int col = 0; col < feats->ncols; ++col)
      feats->val[iftGetMatrixIndex(feats, col, row)] /= sum;
  }
}

// ---------- Public Methods Implementation ----------

iftMatrix * iftComputeSuperpixelFeaturesByColorSpaceMedoid(  iftImage *superpixelLabelMap,   iftImage *img, iftColorSpace colorSpace)
{
  iftMImage *mimg = iftImageToMImage(img, colorSpace);
  iftMatrix *res = iftComputeSuperpixelFeaturesByMImageMedoid(mimg, superpixelLabelMap);
  iftDestroyMImage(&mimg);
  return res;
}

iftMatrix * iftComputeSuperpixelFeaturesByMImageMedoid(  iftMImage *mimg,   iftImage *superpixelLabelMap)
{
  assert(mimg != NULL && superpixelLabelMap != NULL);
  assert(mimg->xsize == superpixelLabelMap->xsize);
  assert(mimg->ysize == superpixelLabelMap->ysize);
  assert(mimg->zsize == superpixelLabelMap->zsize);
  // TODO assert(check label map consistency) 

  // Piggyback from the mean calculation
  iftMatrix *mx = iftComputeSuperpixelFeaturesByMImageMean(mimg, superpixelLabelMap);

  // Keep track of currently known closest pixel for each superpixel
  int *medoidCandidate = iftAllocIntArray(mx->nrows);
  float *candidateDist = iftAllocFloatArray(mx->nrows);
  for (int i = 0; i < mx->nrows; ++i) {
    medoidCandidate[i] = -1;
    candidateDist[i] = FLT_MAX;
  }

  // Try to update closest pixel for each pixel
  for (int p = 0; p < mimg->n; ++p) {
    int row = superpixelLabelMap->val[p] - 1;
    if (row < 0)
      continue;
    // Compute distance from its superpixel mean
    float dist = 0.0;
    for (int f = 0; f < mimg->m; ++f) {
      float a = mimg->val[p][f];
      float b = mx->val[iftGetMatrixIndex(mx,f,row)];
      dist += (a-b)*(a-b);
    }
    // Update on improvement
    if (dist < candidateDist[row]) {
      medoidCandidate[row] = p;
      candidateDist[row] = dist;
    }
  }

  // Overwrite mean matrix with medoids to avoid new allocation
  for (int row = 0; row < mx->nrows; ++row) {
    for (int f = 0; f < mimg->m; ++f) {
      float feat = mimg->val[medoidCandidate[row]][f];
      mx->val[iftGetMatrixIndex(mx,f,row)] = feat;
    }
  }

  free(medoidCandidate);
  free(candidateDist);
  return(mx);
}

iftMatrix * iftComputeSuperpixelFeaturesByColorBoVW(  iftMImage *mimg,   iftImage *superpixelLabelMap,   iftBoVW *bag, int k)
{
  ssygvd_(NULL, NULL, NULL, NULL, NULL,
      NULL, NULL, NULL, NULL, NULL,
      NULL, NULL, NULL, NULL);
  assert(mimg != NULL && superpixelLabelMap != NULL);
  assert(mimg->xsize == superpixelLabelMap->xsize);
  assert(mimg->ysize == superpixelLabelMap->ysize);
  assert(mimg->zsize == superpixelLabelMap->zsize);
  // TODO assert(check label map consistency) 
  assert(mimg->m == bag->nFeats);

  int nSuperpixels = iftMaximumValue(superpixelLabelMap);
  float *featVector = iftAllocFloatArray(mimg->m);
  iftMatrix *mx = iftCreateMatrix(bag->nBins, nSuperpixels); 

  // Assign one pixel at a time
  for (int p = 0; p < mimg->n; ++p) {
    int row = superpixelLabelMap->val[p] - 1;
    if (row < 0)
      continue;

    // Copy mimg pixel feature vector into float array for convenience
    for (int f = 0; f < mimg->m; ++f)
      featVector[f] = mimg->val[p][f]; 

    // Assign pixel feature vector to histogram
    iftBoVWSoftAssignment(bag, featVector, iftMatrixRowPointer(mx, row), k);
  }

  // Normalize histograms
  for (int row = 0; row < mx->nrows; ++row) {
    float histogramTotal = 0.0;
    // Naive float accumulation, O(n) error
    for (int col = 0; col < mx->ncols; ++col)
      histogramTotal += mx->val[iftGetMatrixIndex(mx,col,row)];
    for (int col = 0; col < mx->ncols; ++col)
      mx->val[iftGetMatrixIndex(mx,col,row)] /= histogramTotal;
  }

  free(featVector);
  return mx;
}

iftMatrix * iftComputeSuperpixelFeaturesByRegionCovarianceMatrix(  iftImage *superpixelLabelMap,   iftMatrix *pixelFeats)
{
  assert(superpixelLabelMap->n == pixelFeats->nrows);

  // Compute superpixel information
  int nSuperpixels = iftMaximumValue(superpixelLabelMap);
  int *superpixelSize = iftAllocIntArray(nSuperpixels);
  for (int p = 0; p < superpixelLabelMap->n; ++p) {
    int label = superpixelLabelMap->val[p] - 1;
    if (label < 0)
      continue;
    superpixelSize[label] += 1;
  }

  // Build a dataset of pixel features for each superpixel
  iftDataSet **Zr = iftAlloc(nSuperpixels, sizeof(*Zr));
  for (int i = 0; i < nSuperpixels; ++i)
    Zr[i] = iftCreateDataSet(superpixelSize[i], pixelFeats->ncols);
  int *spIndex = iftAllocIntArray(nSuperpixels);
  for (int row = 0; row < pixelFeats->nrows; ++row) {
    int label = superpixelLabelMap->val[row] - 1;
    if (label < 0)
      continue;
    for (int f = 0; f < pixelFeats->ncols; ++f) {
      Zr[label]->sample[spIndex[label]].feat[f] =
        pixelFeats->val[iftGetMatrixIndex(pixelFeats,f,row)];
    }
    Zr[label]->sample[spIndex[label]].status = IFT_TRAIN; 
    Zr[label]->ntrainsamples += 1;
    spIndex[label] += 1;
  }

  // Build covariance matrix for all superpixels
  int covMxSize = pixelFeats->ncols * pixelFeats->ncols;
  iftMatrix *res = iftCreateMatrix(covMxSize, nSuperpixels);
  for (int i = 0; i < nSuperpixels; ++i) {
    iftDataSet *Zrc = iftCentralizeDataSet(Zr[i]);
    iftMatrix *covMx = iftDatasetCovarianceMatrix(Zrc);
    assert(covMx->n == covMxSize); // sanity test
    // We read covariance matrix as 1D array
    for (int f = 0; f < covMxSize; ++f)
      res->val[iftGetMatrixIndex(res,f,i)] = covMx->val[f];
    iftDestroyDataSet(&(Zr[i]));
    iftDestroyDataSet(&Zrc);
    iftDestroyMatrix(&covMx);
  }
  
  // Clean up
  free(Zr);
  free(superpixelSize);
  free(spIndex);

  return res;
}

iftMatrix * iftComputeSuperpixelFeaturesByColorHistogram(  iftImage *img,   iftImage *labelMap, int binsPerColor)
{
  int nSuperpixels = iftMaximumValue(labelMap);
  int nDims = iftIsColorImage(img) ? 3 : 1;
  int nBins = lround(pow(binsPerColor, nDims));
  iftMatrix *feats = iftCreateMatrix(nBins, nSuperpixels);
  iftImage *binImage = iftColorHistogramAssignPixelBin(img, binsPerColor);

  for (int p = 0; p < labelMap->n; ++p) {
    int label = labelMap->val[p] - 1;
    iftMatrixRowPointer(feats, label)[binImage->val[p]] += 1.0;
  }

  iftDestroyImage(&binImage);
  return feats;
}

iftMatrix * iftComputeSuperpixelFeaturesByBICHistogram(  iftImage *img,   iftImage *labelMap, int binsPerColor)
{
  int nSuperpixels = iftMaximumValue(labelMap);
  int nDims = iftIsColorImage(img) ? 3 : 1;
  int nBins = lround(pow(binsPerColor, nDims));
  iftMatrix *feats = iftCreateMatrix(nBins * 2, nSuperpixels);
  iftImage *binImage = iftColorHistogramAssignPixelBin(img, binsPerColor);
  iftImage *mask = iftBICMask(binImage);
  for (int p = 0; p < labelMap->n; ++p) {
    int offset = (mask->val[p] > 0 ? nBins : 0);
    int label = labelMap->val[p] - 1;
    iftMatrixRowPointer(feats, label)[binImage->val[p] + offset] += 1.0;
  }

  iftDestroyImage(&binImage);
  iftDestroyImage(&mask);
  return feats;
}

iftMatrix * iftComputeSuperpixelFeaturesByNeighborHistogram(  iftSuperpixelIGraph *igraph, iftMatrix *spHistFeats, int neighDist)
{
  iftMatrix *res = iftCreateMatrix(spHistFeats->ncols, spHistFeats->nrows);

  int *spMarker = iftAllocIntArray(igraph->nNodes);
  for (int node = 0; node < igraph->nNodes; ++node) {
    int mark = node + 1; // Color that implicitly resets for each node
    iftSet *spSet = NULL;
    iftInsertSet(&spSet, node);
    spMarker[node] = mark;

    // BFS-like search for the neighbors at specified distance
    for (int dist = 0; dist < neighDist; ++dist) {
      iftSet *auxSet = NULL;
      while(spSet != NULL) {
        int s = iftRemoveSet(&spSet);
        iftSet *adj = igraph->nodeAdj[s];
        for(; adj != NULL; adj = adj->next) {
          int t = adj->elem;
          if (spMarker[t] < mark) {
            spMarker[t] = mark;
            iftInsertSet(&auxSet, t);
          }
        }
      }
      spSet = auxSet;
    }

    // Add neighbor histogram counts to the result matrix
    float *nodeFeat = iftMatrixRowPointer(res, node);
    while (spSet != NULL) {
      int s = iftRemoveSet(&spSet);
      float *neighFeat = iftMatrixRowPointer(spHistFeats, s);
      for (int col = 0; col < res->ncols; ++col)
        nodeFeat[col] += neighFeat[col];
    }
  }

  return res;
}

iftMatrix * iftComputeSuperpixelFeaturesByContextualColorHistogram(  iftSuperpixelIGraph *igraph,   iftImage *img, int binsPerColor, int order)
{
  iftMatrix *baseHist = iftComputeSuperpixelFeaturesByColorHistogram(img, igraph->refImg, binsPerColor);
  iftMatrix *res = iftCopyMatrix(baseHist);
  iftNormalizeSuperpixelFeatMatrix(res);
  for (int i = 0; i < order; ++i) {
    iftMatrix *neighHist = iftComputeSuperpixelFeaturesByNeighborHistogram(igraph, baseHist, i+1);
    iftNormalizeSuperpixelFeatMatrix(neighHist);
    iftMatrix *aux = iftConcatenateSuperpixelFeatMatrix(res, neighHist);
    iftSwap(aux, res);
    iftDestroyMatrix(&aux);
  }

  iftDestroyMatrix(&baseHist);
  return res;
}

iftMatrix * iftComputeSuperpixelFeaturesByContextualBICHistogram(  iftSuperpixelIGraph *igraph,   iftImage *img, int binsPerColor, int order)
{
  iftMatrix *baseHist = iftComputeSuperpixelFeaturesByBICHistogram(img, igraph->refImg, binsPerColor);
  iftMatrix *res = iftCopyMatrix(baseHist);
  iftNormalizeSuperpixelFeatMatrix(res);
  for (int i = 0; i < order; ++i) {
    iftMatrix *neighHist = iftComputeSuperpixelFeaturesByNeighborHistogram(igraph, baseHist, i+1);
    iftNormalizeSuperpixelFeatMatrix(neighHist);
    iftMatrix *aux = iftConcatenateSuperpixelFeatMatrix(res, neighHist);
    iftSwap(aux, res);
    iftDestroyMatrix(&aux);
  }

  iftDestroyMatrix(&baseHist);
  return res;
}
