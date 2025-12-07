#include "iftBoVW.h" 
iftSet *resRoots = NULL; // Temporary, for validation

// ---------- Private Methods Declaration ----------
/**
 * @brief Creates new BoVW with the input domain.
 * @author Felipe Lemes Galvao
 * @date June 27, 2017
 *
 * @details The returned \c iftBoVW is allocated on the heap and owned
 *   by the caller. Only the domain values from \c iftBoVW are
 *   initialized.
 * @see iftDestroyBoVW()
 *
 * @param[in] nFeats Number of features for each cluster prototype.
 * @param[in] nBins Number of histogram bins, each represented by a
 *            cluster prototype.
 * @return The new BoVW.
 *
 * @pre Both input parameters are positive.
 */
iftBoVW * iftCreateBoVW(int nBins, int nFeats);

/**
 * @brief Computes the distance of a feature vector to all BoVW words.
 * @author Felipe Lemes Galvao
 * @date June 28, 2017
 *
 * @details The computed distance omits the sqrt computation as we
 *   are only interested in the ordering. The returned \c float array
 *   is allocated on the heap and owned by the caller.
 *
 * @param[in] bag The BoVW descriptor.
 * @param[in] input The feature vector.
 * @return The array of size \c bag->nBins with the distances.
 *
 * @pre All parameters are not NULL and have matching dimensions.
 */
float * iftBoVWInputDistance(  iftBoVW *bag,   float *input);

// ---------- Private Methods Implementation ----------
iftBoVW * iftCreateBoVW(int nBins, int nFeats)
{
  assert(nBins > 0);
  assert(nFeats > 0);
  iftBoVW *bag = (iftBoVW *) iftAlloc(1, sizeof(*bag));

  bag->nFeats = nFeats;
  bag->nBins = nBins;
  bag->feats = iftCreateMatrix(nFeats, nBins);
  // iftSetBoVWDefaultHistogramAssignment(bag)

  return bag;
}

float * iftBoVWInputDistance(  iftBoVW *bag,   float *input)
{
  assert(bag != NULL && input != NULL);
  // We don't have array size to check matching dimensions

  float *distances = iftAllocFloatArray(bag->nBins);
  for (int bin = 0; bin < bag->nBins; ++bin) {
    for (int f = 0; f < bag->nFeats; ++f) {
      float a = input[f];
      float b = bag->feats->val[iftGetMatrixIndex(bag->feats,f,bin)];
      distances[bin] += (a-b)*(a-b);
    }
  }

  return distances;
}

// ---------- Public Methods Implementation ----------
iftBoVW * iftCreateBoVWByOPFClustering(iftMatrix *mx, float kmaxPercent, iftKnnGraphCutFun graphCutFun)
{
  // Convert data matrix of representative values into a dataset
  iftDataSet *Z = iftFeatureMatrixToDataSet(mx);
  Z->ntrainsamples = Z->nsamples;
  for (int s = 0; s < Z->nsamples; ++s)
    Z->sample[s].status = IFT_TRAIN;

  // Compute Unsupervised OPF over superpixel prototypes
  int kmax = ((float) Z->ntrainsamples) * kmaxPercent;
  iftKnnGraph *graph = iftCreateKnnGraph(Z, kmax);
  int nClusters = iftUnsupTrain(graph, graphCutFun);

  // Create BoVW with resulting OPF roots as prototypes
  iftBoVW *bag = iftCreateBoVW(nClusters, Z->nfeats);
  iftSet *roots = iftGetKnnRootSamples(graph);
  resRoots = iftSetCopy(roots); // Temporary for debug
  int binPos = 0;
  while (roots != NULL) {
    int r = iftRemoveSet(&roots);
    for (int f = 0; f < Z->nfeats; ++f)
      bag->feats->val[iftGetMatrixIndex(bag->feats,f,binPos)] =
        Z->sample[r].feat[f];
    binPos += 1;
  }

  iftDestroyKnnGraph(&graph);
  iftDestroyDataSet(&Z);
  return bag;
}

iftBoVW * iftCreateBoVWByKmeans(iftMatrix *mx, int k, int maxIters, float minImprovement)
{
  // Convert data matrix of representative values into a dataset
  iftDataSet *Z = iftFeatureMatrixToDataSet(mx);
  Z->ntrainsamples = Z->nsamples;
  for (int s = 0; s < Z->nsamples; ++s)
    Z->sample[s].status = IFT_TRAIN;

  // Compute K-means
  iftDataSet *Zk = iftKmeansInitCentroidsFromSamples(Z, k);
  iftKmeansRun(0, Z, &Zk, maxIters, minImprovement);

  // Create BoVW with cluster means as words
  assert(k == Zk->nsamples); // sanity test
  iftBoVW *bag = iftCreateBoVW(k, Z->nfeats);
  for (int s = 0; s < k; ++s)
    for (int f = 0; f < Z->nfeats; ++f)
      bag->feats->val[iftGetMatrixIndex(bag->feats,f,s)] =
        Zk->sample[s].feat[f];

  iftDestroyDataSet(&Z);
  iftDestroyDataSet(&Zk);
  return bag;
}

void iftDestroyBoVW(iftBoVW **bag)
{
  if (bag == NULL || *bag == NULL)
    return;

  iftBoVW * tmp = *bag;
  iftDestroyMatrix(&(tmp->feats));
  free(tmp);
  *bag = NULL;
}

void iftBoVWHardAssignment(  iftBoVW *bag,   float *input, float *histogram)
{
  assert(bag != NULL && input != NULL && histogram != NULL);
  // We do not have array sizes to check matching dimensions

  // Compute all distances
  float *distances = iftBoVWInputDistance(bag, input);

  // Find minimum value
  float minDist = FLT_MAX;
  int minIndex = -1;
  for (int i = 0; i < bag->nBins; ++i) {
    if (distances[i] < minDist) {
      minDist = distances[i];
      minIndex = i;
    }
  }

  // Make hard assignment
  histogram[minIndex] += 1.0;

  free(distances);
}

void iftBoVWSoftAssignment(  iftBoVW *bag,   float *input, float *histogram, int k)
{
  assert(bag != NULL && input != NULL && histogram != NULL);
  assert(k > 0);
  // We do not have array sizes to check matching dimensions

  // Special case for k = 1 (hard assignment)
  if (k == 1) {
    iftBoVWHardAssignment(bag, input, histogram);
    return;
  }

  // Compute distance to all visual words
  float *distances = iftBoVWInputDistance(bag, input);

  // Use sorting algorithm to find closest words
  int *index = iftAllocIntArray(bag->nBins);
  for (int i = 0; i < bag->nBins; ++i)
    index[i] = i;
  iftFHeapSort(distances, index, bag->nBins, IFT_INCREASING); 
  
  // Calculate word normalization weight
  double kSum = 0.0;
  for (int i = 0; i < k; ++i)
    kSum += distances[i];

  // Make soft assignment. We add a total of 1.0 to the histogram,
  //  distributed over the k closest words.
  for (int i = 0; i < k; ++i)
    histogram[index[i]] += (distances[i] / kSum);

  free(distances);
  free(index);
}
