/**
 * @file
 * @brief Implementation of Nakamura and Hong (2017) segmentation.
 * @author Felipe Lemes Galvao
 *
 *   See Nakamura, K. & Hong, B.W. JEI (2017).
 *   This program computes a single hierarchy level at a time. In the first level,
 *     the initial regular partition must be given as input (honeycomb/voronoi
 *     partition can be obtained with iftRISF_segmentation).
 *   We use our RAG struct for convenience, the method does not rely on the graph
 *     adjacency.
 */

#include <ift.h>
#include "iftSuperpixelFeatures.h"
#include "iftExperimentUtility.h"

#define INPUT_ARG "--input-image"
#define OUTPUT_ARG "--output-image"
#define NSP_ARG "--target-nsp"
#define TAL_ARG "--tal"
#define THETA_ARG "--theta"
#define PART_ARG "--prev-partition"
#define FEAT_ARG "--superpixel-feat"
#define PRINT_ARG "--print-opt"
#define PREVSEG_ARG "--prev-seg"
#define MASK_ARG "--mask"
#define FORMAT_ARG "--output-format"

// iftL1Norm divides result by n for some reason
float nkL1Norm(float *f1, float *f2, float *alpha, int n) 
{
  double distance = 0.0;
  for (int i = 0; i < n; ++i)
    distance += fabs(f1[i] - f2[i]);

  return ((float)distance);
}

iftDict *iftGetArguments(int argc, const char *argv[]);
iftImage * iftSuperpixelSegmentationByNakamuraPx(iftSuperpixelIGraph *igraph, iftImage *initPartition);
iftImage * iftSuperpixelSegmentationByNakamuraSp(iftSuperpixelIGraph *igraph, iftImage *initPartition);
void UpdateClusterCentersSp(iftSuperpixelIGraph *igraph, iftMatrix *labelMx);
void UpdateClusterCentersPx(iftSuperpixelIGraph *igraph, int *labelArr);
int set_find(int *parent, int node);
void set_union(int *parent, int *rank, int a, int b);
void nkComputeNodeToPrototypeDist(iftSuperpixelIGraph *igraph, iftMatrix *featDistToPrototype, iftMatrix *spaceDistToPrototype, int node, int c, float normalizationFactor);
void nkInitClusters(iftSuperpixelIGraph *igraph, iftImage *initPartition, iftMatrix *labelMx, int nClusters); 
void MergeOverlappingClusters(iftSuperpixelIGraph *igraph, iftMatrix **labelMx, int *pNClusters);
void UpdateClusterCentersPx(iftSuperpixelIGraph *igraph, int *labelArr);
iftImage *iftBuildGridInitPartition(iftSuperpixelIGraph *igraph, int nSp);
iftImage * iftSuperpixelSegmentationByNakamuraSpStd(iftSuperpixelIGraph *igraph, iftImage *initPartition);
void UpdateClusterCentersSpNoMerge(iftSuperpixelIGraph *igraph, int *labelArr);
iftSet * nkGetPartitionCenters(iftImage *part);
iftImage * iftSuperpixelSegmentationByNakamuraSpMerge(iftSuperpixelIGraph *igraph, iftImage *initPartition);

int main(int argc, char* argv[])
{
  iftDict* args = iftGetArguments(argc, (const char **) argv);
  char *tmpPath;

  int printOpt = (iftDictContainKey(PRINT_ARG, args, NULL)) ?
    iftGetLongValFromDict(PRINT_ARG, args) : 1;
  bool isHierarchic = iftDictContainKey(PREVSEG_ARG, args, NULL);
  bool hasMask = iftDictContainKey(MASK_ARG, args, NULL);

  // Load Image
  tmpPath = iftGetStrValFromDict(INPUT_ARG, args);
  iftImage *img = iftReadImageByExt(tmpPath);
  free(tmpPath);
  bool is3D = iftIs3DImage(img);

  if (hasMask)
    iftError("main", "Operation with mask not fully implemented yet.");
  if (is3D)
    iftError("main", "Operation in 3D not fully implemented yet.");

  // Prepare igraph reference image
  iftImage *refImg = NULL;
  if (isHierarchic) {
    // Load pre-computed superpixel segmentation (previous RISF level)
    tmpPath = iftGetStrValFromDict(PREVSEG_ARG, args);
    refImg = iftReadImageByExt(tmpPath);
    free(tmpPath);
    if (hasMask) {
      tmpPath = iftGetStrValFromDict(MASK_ARG, args);
      iftImage *mask = iftReadImageByExt(tmpPath);
      free(tmpPath);

      iftDestroyImage(&mask);
      printf("Hierarchy + mask not implemented yet.\n");
      return -1;
    }
  } else {
    // Make superpixel graph act as a pixel-based one
    if (!hasMask) {
      refImg = iftSelectImageDomain(img->xsize, img->ysize, img->zsize);
      for (int i = 0; i < refImg->n; ++i)
        refImg->val[i] = i + 1;
    } else {
      tmpPath = iftGetStrValFromDict(MASK_ARG, args);
      refImg = iftReadImageByExt(tmpPath);
      free(tmpPath);
      int label = 1;
      for (int i = 0; i < refImg->n; ++i) {
        if (refImg->val[i] > 0)
          refImg->val[i] = label++;
      }
    }
  }

  // Build igraph
  iftSuperpixelIGraph *igraph = iftInitSuperpixelIGraph(refImg);
  iftAdjRel *A = is3D ? iftSpheric(1.0) : iftCircular(1.0);
  // Abuse of notation, we do not use an actual graph representation
  if (isHierarchic)
    iftSetSuperpixelIGraphExplicitRAG(igraph, A);
  else
    iftSetSuperpixelIGraphImplicitAdjacency(igraph, A);

  // Set superpixel parametric features
  float alpha = (iftDictContainKey(THETA_ARG, args, NULL)) ?
    iftGetDblValFromDict(THETA_ARG, args) : 16;
  float beta =  (iftDictContainKey(TAL_ARG, args, NULL)) ? 
    iftGetDblValFromDict(TAL_ARG, args) : 0.0f;
  int featOpt = (iftDictContainKey(FEAT_ARG, args, NULL)) ?
    iftGetLongValFromDict(FEAT_ARG, args) : 1;
  iftMatrix *pFeats = NULL;
  switch (featOpt) {
    case 1: // Color mean
    default:
      {
        if (!isHierarchic) {
          if (iftIsColorImage(img))
            pFeats = iftComputeSuperpixelFeaturesByColorSpaceMean(igraph->refImg, img, LABNorm_CSPACE);  
          else
            pFeats = iftComputeSuperpixelFeaturesByColorSpaceMean(igraph->refImg, img, GRAYNorm_CSPACE);  
          // Build dataset to apply whitening transform
          iftDataSet *Z = iftFeatureMatrixToDataSet(pFeats);
          Z->ntrainsamples = Z->nsamples;
          iftDataSet *Zw = iftWhiteningTransform(Z);
          iftDestroyMatrix(&pFeats);
          pFeats = iftDataSetToFeatureMatrix(Zw);
          iftDestroyDataSet(&Z);
          iftDestroyDataSet(&Zw);
        } else {
          iftImage *pxMap = iftCopyImage(igraph->refImg);
          for (int i = 0; i < pxMap->n; ++i)
            pxMap->val[i] = i + 1;
          iftMatrix *pxPFeats;
          if (iftIsColorImage(img))
            pxPFeats = iftComputeSuperpixelFeaturesByColorSpaceMean(pxMap, img, LABNorm_CSPACE);  
          else
            pxPFeats = iftComputeSuperpixelFeaturesByColorSpaceMean(pxMap, img, GRAYNorm_CSPACE);  
          iftDataSet *Z = iftFeatureMatrixToDataSet(pxPFeats);
          Z->ntrainsamples = Z->nsamples;
          iftDataSet *Zw = iftWhiteningTransform(Z);
          iftDestroyMatrix(&pxPFeats);
          pxPFeats = iftDataSetToFeatureMatrix(Zw);
          iftDestroyDataSet(&Z);
          iftDestroyDataSet(&Zw);
          // Compute median - re-use seed calculation function
          iftSuperpixelIGraph *auxIgraph = iftInitSuperpixelIGraph(pxMap);
          iftSetSuperpixelIGraphFeatures(auxIgraph, pxPFeats, nkL1Norm, alpha, beta);
          auxIgraph->seedFeats = iftCreateMatrix(auxIgraph->feats->ncols, igraph->nNodes);
          auxIgraph->seedPos = iftCreateMatrix(auxIgraph->pos->ncols, igraph->nNodes);
          UpdateClusterCentersPx(auxIgraph, igraph->refImg->val);
          //UpdateClusterCentersSpNoMerge(auxIgraph, igraph->refImg->val);
          pFeats = iftCopyMatrix(auxIgraph->seedFeats);
          iftDestroyImage(&pxMap);
          iftDestroyMatrix(&pxPFeats);
          iftDestroySuperpixelIGraph(&auxIgraph);
        }
        iftSetSuperpixelIGraphFeatures(igraph, pFeats, nkL1Norm, alpha, beta);
        //iftSetSuperpixelIGraphFeatures(igraph, pFeats, iftDistance1, alpha, beta);
      }
      break;
  }

  // Get initial partition
  iftImage *initPartition = NULL;
  if (iftDictContainKey(PART_ARG, args, NULL)) {
    tmpPath = iftGetStrValFromDict(PART_ARG, args);
    initPartition = iftReadImageByExt(tmpPath);
    free(tmpPath);
  } else if (iftDictContainKey(NSP_ARG, args, NULL) && igraph->type == EXPLICIT) {
    int nSp = iftGetLongValFromDict(NSP_ARG, args);
    //initPartition = iftBuildMetisInitPartition(igraph, nSp);
    initPartition = iftBuildGridInitPartition(igraph, nSp);
  } else if (!isHierarchic) {
    iftError("main", "Pixel level needs initialization.\n");
  }

  // Compute segmentation
  timer *t1 = iftTic();
  iftImage *superpixelLabelMap = isHierarchic ?
    iftSuperpixelSegmentationByNakamuraSpMerge(igraph, initPartition) :
    iftSuperpixelSegmentationByNakamuraPx(igraph, initPartition);
  timer *t2 = iftToc();

  // Time info
  if (printOpt == 1)
    printf("Processing time: %fms\n", iftCompTime(t1, t2));
  else if (printOpt == 2)
    printf("%f", iftCompTime(t1, t2));

  // Write segmentation result to file
  tmpPath = iftGetStrValFromDict(OUTPUT_ARG, args);
  int outputFormat = (iftDictContainKey(FORMAT_ARG, args, NULL)) ?
    iftGetLongValFromDict(FORMAT_ARG, args) : 0;
  switch (outputFormat) {
    case 1:
      iftWriteImageP2(superpixelLabelMap, tmpPath);
      break;
    case 2:
      iftWriteImageP5(superpixelLabelMap, tmpPath);
      break;
    default:
    case 0:
      iftWriteImageByExt(superpixelLabelMap, tmpPath);
      break;
  }
  free(tmpPath);

  // Clean up
  iftDestroyImage(&img);
  iftDestroyImage(&refImg);
  iftDestroySuperpixelIGraph(&igraph);
  iftDestroyMatrix(&pFeats);
  iftDestroyImage(&superpixelLabelMap);

  return 0;
}

#define FEAT_HELP "Superpixel features:\n\
    1 = LAB mean (DEFAULT)"

#define PRINT_HELP "Printing options:\n\
    0 = Nothing is printed\n\
    1 = Human-readable basic stats (DEFAULT)\n\
    2 = CSV formatted stats"

#define OUTPUT_FORMAT_HELP "Label map output format:\n\
    0 = Deduce by extension (DEFAULT)\n\
    1 = PGM P2 - ASCII\n\
    2 = PGM P5 - binary"

iftDict *iftGetArguments(int argc, const char *argv[]) {
  iftCmdLineOpt cmd_line_opts[] = {
    {.short_name = "-i", .long_name = INPUT_ARG, .has_arg = true,
      .arg_type = IFT_STR_TYPE, .required = true, 
      .help = "Input image path."},
    {.short_name = "-o", .long_name = OUTPUT_ARG, .has_arg = true,
      .arg_type = IFT_STR_TYPE, .required = true,
      .help = "Output label map image."},
    {.short_name = "-n", .long_name = NSP_ARG, .has_arg = true,
      .arg_type = IFT_LONG_TYPE, .required = false,
      .help = "Target number of regions. Ignored if -s is supplied."},
    {.short_name = "-a", .long_name = THETA_ARG, .has_arg = true,
      .arg_type = IFT_DBL_TYPE, .required = false,
      .help = "Lower is less compact. Default = 16"},
    {.short_name = "-b", .long_name = TAL_ARG, .has_arg = true,
      .arg_type = IFT_DBL_TYPE, .required = false,
      .help = "Tolerance for merging clusters. Default is 0.04 * scale."},
    {.short_name = "-s", .long_name = PART_ARG, .has_arg = true,
      .arg_type = IFT_STR_TYPE, .required = false,
      .help = "Pre-existing partition for initialization."},
    {.short_name = "-f", .long_name = FEAT_ARG, .has_arg = true,
      .arg_type = IFT_LONG_TYPE, .required = false,
      .help = FEAT_HELP},
    {.short_name = "-p", .long_name = PRINT_ARG, .has_arg = true,
      .arg_type = IFT_LONG_TYPE, .required = false,
      .help = PRINT_HELP},
    {.short_name = "-l", .long_name = PREVSEG_ARG, .has_arg = true,
      .arg_type = IFT_STR_TYPE, .required = false,
      .help = "Previous segmentation label map for higher levels."},
    {.short_name = "-m", .long_name = MASK_ARG, .has_arg = true,
      .arg_type = IFT_STR_TYPE, .required = false,
      .help = "Mask for constrained domain segmentation."},
    {.short_name = "-x", .long_name = FORMAT_ARG, .has_arg = true,
      .arg_type = IFT_LONG_TYPE, .required = false,
      .help = OUTPUT_FORMAT_HELP}
  };
  int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

  char program_description[IFT_STR_DEFAULT_SIZE] = 
    "This is a general demo to run RISF methods.";

  // Parser Setup
  iftCmdLineParser *parser =
    iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
  iftDict *args = iftParseCmdLine(argc, argv, parser);
  iftDestroyCmdLineParser(&parser);

  return args;
}

iftImage * iftSuperpixelSegmentationByNakamuraPx(iftSuperpixelIGraph *igraph, iftImage *initPartition)
{
  if (initPartition == NULL)
    iftError("iftSuperpixelSegmentationByNakamuraPx", "An initial partition is required");

  // Data
  int nClusters = iftMaximumValue(initPartition);
  int *labelArr = iftAllocIntArray(igraph->nNodes);
  float *minDist = iftAllocFloatArray(igraph->nNodes);

  // Values that are stored across iterations need initialization beforehand
  for (int i = 0; i < igraph->nNodes; ++i) {
    labelArr[i] = -1;
    minDist[i] = IFT_INFINITY_FLT;
  }
  
  // -- Cluster initialization
  igraph->seedFeats = iftCreateMatrix(igraph->feats->ncols, nClusters);
  igraph->seedPos = iftCreateMatrix(igraph->pos->ncols, nClusters);
  // This methods only picks the central pixel and copies their features
  iftSet *seeds = nkGetPartitionCenters(initPartition);
  while (seeds != NULL) {
    int p = iftRemoveSet(&seeds);
    int label = initPartition->val[p] - 1;
    int node = igraph->refImg->val[p] - 1;
    for (int f = 0; f < igraph->seedFeats->ncols; ++f)
      iftMatrixRowPointer(igraph->seedFeats, label)[f] =
        iftMatrixRowPointer(igraph->feats, node)[f];
    for (int f = 0; f < igraph->seedPos->ncols; ++f)
      iftMatrixRowPointer(igraph->seedPos, label)[f] =
        iftMatrixRowPointer(igraph->pos, node)[f];
  }
  // This method uses the entire partition as an actual segmentation
  /*for (int p = 0; p < initPartition->n; ++p) {
    int label = initPartition->val[p] - 1;
    int node = igraph->refImg->val[p] - 1;
    if (label < 0 || node < 0)
      continue;
    labelArr[node] = label;
  }
  UpdateClusterCentersPx(igraph, labelArr);*/
  
  // -- Main loop
  int iterCount = 0;
  do {
    printf("Running iter %d\n", iterCount++);
    bool hasChanged = false;

    // Regularizes lambda to be invariant to the target number of superpixels
    float normalizationFactor = sqrt((IFT_PI * ((float)nClusters)) / ((float)igraph->refImg->n)) / logf((float)nClusters);

    // -- Find closest prototype to each node
    for (int c = 0; c < nClusters; ++c) {
      // Find SLIC-based search window
      int step = (int) ceil(sqrtf(((float)igraph->refImg->n) / ((float)nClusters)));
      iftVoxel cPx = {
        .x = (int) roundf(iftMatrixRowPointer(igraph->seedPos, c)[0]),
        .y = (int) roundf(iftMatrixRowPointer(igraph->seedPos, c)[1]),
        .z = 0};
      iftVoxel u = {0, 0, 0};
      for (u.y = cPx.y - step; u.y < cPx.y + step; ++(u.y)) {
        for (u.x = cPx.x - step; u.x < cPx.x + step; ++(u.x)) {
          if (!iftValidVoxel(igraph->refImg, u))
            continue;
          int node = iftGetVoxelIndex(igraph->refImg, u);

          // Base distances
          float featDist = iftSpIGraphNodeToSeedFeatDist(igraph, node, c);
          float spaceDist = iftSpIGraphNodeToSeedSpaceDist(igraph, node, c);

          // Lambda is the compactness parameter defined in an adaptative manner
          // Should be around ~0.75 withut the adaptative value
          float lambda = exp(-featDist * logf((float)nClusters)) * normalizationFactor;
          float dist = featDist + lambda * spaceDist;
          // Note that minDist is stored across iterations to force convergence
          if (dist < minDist[node]) {
            hasChanged = true;
            minDist[node] = dist;
            labelArr[node] = c;
          }
        }
      }
    }

    // Check for convergence
    if (!hasChanged)
      break;

    // Update cluster centers based on label matrix
    UpdateClusterCentersPx(igraph, labelArr);
    //UpdateClusterCentersSpNoMerge(igraph, labelArr);
  } while (true);

  // Build initial label map
  iftImage *labelMap = iftCopyImage(igraph->refImg);
  for (int p = 0; p < labelMap->n; ++p) {
    int node = labelMap->val[p] - 1;
    if (node < 0)
      continue;
    labelMap->val[p] = labelArr[node]; 
  }

  // SLIC-like post-processing to fix unconnected regions & remove small ones
  printf("Post-processing\n");
  int minSize = igraph->type == IMPLICIT ? ((initPartition->n / nClusters) / 2): 0;
  iftImage *res = iftForceLabelMapConnectivity(labelMap, minSize);

  // Clean-up
  free(labelArr);
  iftDestroyImage(&labelMap);
  free(minDist);

  return res;
}

iftImage * iftSuperpixelSegmentationByNakamuraSpMerge(iftSuperpixelIGraph *igraph, iftImage *initPartition)
{
  // Input parameters translation
  float theta = igraph->alpha;
  float tal = igraph->beta; // delta tal update should happen externally

  if (tal < IFT_EPSILON && initPartition == NULL)
    iftError("iftSuperpixelSegmentationByNakamuraSp",
        "Either tal > 0 or an initial partition must be provided");

  // Data
  bool canMerge = (tal > IFT_EPSILON);
  int nClusters = (initPartition != NULL)  ? iftMaximumValue(initPartition) : igraph->nNodes;
  // row = node, col = prototype/cluster
  // Do NOT iterate over their dimensions as nClusters can reduce over iterations
  iftMatrix * labelMx = iftCreateMatrix(nClusters, igraph->nNodes);
  iftMatrix * featDistToPrototype = iftCreateMatrix(nClusters, igraph->nNodes);
  iftMatrix * spaceDistToPrototype = iftCreateMatrix(nClusters, igraph->nNodes);
  iftMatrix * finalDistToPrototype = iftCreateMatrix(nClusters, igraph->nNodes);
  float *avgDist = iftAllocFloatArray(igraph->nNodes);
  float *avgFeatDist = iftAllocFloatArray(igraph->nNodes);
  float *avgSpaceDist = iftAllocFloatArray(igraph->nNodes);
  float *avgLambda = iftAllocFloatArray(igraph->nNodes);
  float *minDist = iftAllocFloatArray(igraph->nNodes);
  int *minMap = iftAllocIntArray(igraph->nNodes);

  // Values that are stored across iterations need initialization beforehand
  for (int i = 0; i < labelMx->n; ++i)
    labelMx->val[i] = -1.0f;
  for (int i = 0; i < igraph->nNodes; ++i)
    minDist[i] = IFT_INFINITY_FLT;
  
  // Cluster initialization
  nkInitClusters(igraph, initPartition, labelMx, nClusters);

  // -- Main loop
  int iterCount = 0;
  do {
    printf("Running iter %d\n", iterCount++);
    bool hasChanged = false;

    // -- Pre-compute distances for each node x cluster
    float normalizationFactor = sqrt((IFT_PI * ((float)nClusters)) / ((float)igraph->refImg->n)) / logf((float)nClusters);
    normalizationFactor *= 0.0f;
    for (int node = 0; node < igraph->nNodes; ++node) 
      for (int c = 0; c < nClusters; ++c) 
        nkComputeNodeToPrototypeDist(igraph, featDistToPrototype, spaceDistToPrototype, node, c, normalizationFactor);

    // -- Find closest prototype to each node
    for (int node = 0; node < igraph->nNodes; ++node) {
      avgDist[node] = 0.0f;
      avgFeatDist[node] = 0.0f;
      avgSpaceDist[node] = 0.0f;
      avgLambda[node] = 0.0f;
      for (int c = 0; c < nClusters; ++c) {
        // Lambda is the compactness parameter defined in an adaptative manner
        float lambda = exp(-iftMatrixElem(featDistToPrototype, c, node) *
            logf((float)nClusters) - (theta * tal));
        avgLambda[node] += lambda;
        float dist = iftMatrixElem(finalDistToPrototype, c, node) =
          iftMatrixElem(featDistToPrototype, c, node) 
          + lambda * iftMatrixElem(spaceDistToPrototype, c, node);
        // Note that minDist is stored across iterations to force convergence
        if (dist < minDist[node]) {
          hasChanged = true;
          minDist[node] = dist;
          minMap[node] = c;
        }
        avgDist[node] += dist;
        avgFeatDist[node] += iftMatrixElem(featDistToPrototype, c, node);
        avgSpaceDist[node] += iftMatrixElem(spaceDistToPrototype, c, node);
      }
      avgDist[node] /= nClusters;
      avgFeatDist[node] /= nClusters;
      avgSpaceDist[node] /= nClusters;
      avgLambda[node] /= nClusters;
    }

    float gAvg = 0.0f;
    for (int i = 0; i < igraph->nNodes; ++i)
      gAvg += avgDist[i];
    gAvg /= igraph->nNodes;

    // -- Label nodes
    float epsilon = 0.001f * gAvg;
    int nRounds = 0;
    int mergeCount = 0;
    bool mergeFlag = false;
    while (!mergeFlag) {
      nRounds++;
      for (int node = 0; node < igraph->nNodes; ++node) {
        iftMatrixElem(labelMx, minMap[node], node) = 1.0f;
        if (canMerge) {
          // Epsilon defines the energy differential that allows regions to merge
          // Note that each node can have multiple labels when epsilon > 0
          //float epsilon = (avgDist[node] - minDist[node]) * tal;
          for (int c = 0; c < nClusters; ++c) {
            if (iftMatrixElem(finalDistToPrototype, c, node) < minDist[node] + epsilon) {
              if (iftMatrixElem(labelMx, c, node) < 0) {
                //hasChanged = true;
                if (iftMatrixElem(finalDistToPrototype, c, node) > IFT_EPSILON) {
                  if (iftSetHasElement(igraph->nodeAdj[node], c)) { 
                    mergeFlag = true;
                    printf("Found connected merge (%d,%d) in %d rounds with epsilon = %f iter %d\n", node, c, nRounds, epsilon, iterCount);
                  }
                  mergeCount++;
                }
                iftMatrixElem(labelMx, c, node) = 1.0f;
              }
            }
          }
        }
      }
      epsilon *= 1.001f;
    }
    printf("identified %d one-way merges\n", mergeCount);
    hasChanged = false;

    // Check for convergence
    if (!hasChanged)
      break;

    // Update cluster centers based on label matrix
    UpdateClusterCentersSp(igraph, labelMx);
  } while (true);

  /*printf("\n\n---Final set of prototypes---\n\n");
  for (int c = 0; c < nClusters; ++c) {
    printf("prototype %d = (%f,%f,%f) + (%f,%f)\n", c,
        iftMatrixElem(igraph->seedFeats, 0, c),
        iftMatrixElem(igraph->seedFeats, 1, c),
        iftMatrixElem(igraph->seedFeats, 2, c),
        iftMatrixElem(igraph->seedPos, 0, c),
        iftMatrixElem(igraph->seedPos, 1, c));
  }*/

  if (canMerge)
    MergeOverlappingClusters(igraph, &labelMx, &nClusters);

  printf("Final reduction from %d to %d regions\n", igraph->nNodes, nClusters);

  // Build node to cluster look-up table
  int * nodeToCluster = iftAllocIntArray(igraph->nNodes);
  for (int node = 0; node < igraph->nNodes; ++node) {
    // Find marked cluster
    for (int c = 0; c < nClusters; ++c) {
      if (iftMatrixElem(labelMx, c, node) > 0.0f) {
        nodeToCluster[node] = c;
        break;
      }
    }
  }

  // Build initial label map
  iftImage *labelMap = iftCopyImage(igraph->refImg);
  for (int p = 0; p < labelMap->n; ++p) {
    int node = labelMap->val[p] - 1;
    if (node < 0)
      continue;
    labelMap->val[p] = nodeToCluster[node]; 
  }

  // SLIC-like post-processing to fix unconnected regions & remove small ones
  int minSize = 0;// (igraph->refImg->n / iftMaximumValue(labelMap)) / 2;
  iftImage *res = iftForceLabelMapConnectivity(labelMap, minSize); 
  printf("Post-processing %d to %d regions\n", iftMaximumValue(labelMap), iftMaximumValue(res));

  // Clean-up
  iftDestroyMatrix(&labelMx);
  iftDestroyMatrix(&featDistToPrototype);
  iftDestroyMatrix(&spaceDistToPrototype);
  iftDestroyMatrix(&finalDistToPrototype);
  free(avgDist);
  free(minDist);

  return res;
}

iftImage * iftSuperpixelSegmentationByNakamuraSpStd(iftSuperpixelIGraph *igraph, iftImage *initPartition)
{
  // Input parameters translation
  float theta = igraph->alpha;

  if (initPartition == NULL)
    iftError("iftSuperpixelSegmentationByNakamuraSpStd",
        "Initial partition must be provided.");

  // Data
  int nClusters = iftMaximumValue(initPartition);
  int * labelArr = iftAllocIntArray(igraph->nNodes);
  float *minDist = iftAllocFloatArray(igraph->nNodes);

  if (nClusters >= igraph->nNodes)
    iftError("iftSuperpixelSegmentationByNakamuraSpStd",
        "Initial partition has too many labels.");

  // Values that are stored across iterations need initialization beforehand
  for (int i = 0; i < igraph->nNodes; ++i) {
    labelArr[i] = -1;
    minDist[i] = IFT_INFINITY_FLT;
  }
  
  // Cluster initialization
  for (int p = 0; p < initPartition->n; ++p) {
    int label = initPartition->val[p] - 1;
    int node = igraph->refImg->val[p] - 1;
    if (label < 0 || node < 0)
      continue;
    labelArr[node] = label;
  }

  // Obtain initial prototypes
  igraph->seedFeats = iftCreateMatrix(igraph->feats->ncols, nClusters);
  igraph->seedPos = iftCreateMatrix(igraph->pos->ncols, nClusters);
  UpdateClusterCentersSpNoMerge(igraph, labelArr);

  // -- Main loop
  int iterCount = 0;
  do {
    printf("Running iter %d\n", iterCount++);
    bool hasChanged = false;

    // Regularizes lambda to be invariant to the target number of superpixels
    float normalizationFactor = sqrt((IFT_PI * ((float)nClusters)) / ((float)igraph->refImg->n)) / logf((float)nClusters);

    // -- Find closest prototype to each node
    for (int node = 0; node < igraph->nNodes; ++node) {
      for (int c = 0; c < nClusters; ++c) {
        // Base distances
        float featDist = iftSpIGraphNodeToSeedFeatDist(igraph, node, c);
        float spaceDist = iftSpIGraphNodeToSeedSpaceDist(igraph, node, c);

        // Lambda is the compactness parameter defined in an adaptative manner
        // Should be around ~0.75 withut the adaptative value
        float lambda = exp(-featDist) * normalizationFactor;
        float dist = featDist + lambda * spaceDist;
        // Note that minDist is stored across iterations to force convergence
        if (dist < minDist[node]) {
          hasChanged = true;
          minDist[node] = dist;
          labelArr[node] = c;
        }
      }
    }

    // Check for convergence
    if (!hasChanged)
      break;

    // Update cluster centers based on label matrix
    UpdateClusterCentersSpNoMerge(igraph, labelArr);
  } while (true);

  /*printf("\n\n---Final set of prototypes---\n\n");
  for (int c = 0; c < nClusters; ++c) {
    printf("prototype %d = (%f,%f,%f) + (%f,%f)\n", c,
        iftMatrixElem(igraph->seedFeats, 0, c),
        iftMatrixElem(igraph->seedFeats, 1, c),
        iftMatrixElem(igraph->seedFeats, 2, c),
        iftMatrixElem(igraph->seedPos, 0, c),
        iftMatrixElem(igraph->seedPos, 1, c));
  }*/

  // Build initial label map
  iftImage *labelMap = iftCopyImage(igraph->refImg);
  for (int p = 0; p < labelMap->n; ++p) {
    int node = labelMap->val[p] - 1;
    if (node < 0)
      continue;
    labelMap->val[p] = labelArr[node]; 
  }

  // SLIC-like post-processing to fix unconnected regions & remove small ones
  int minSize = 0;// (igraph->refImg->n / iftMaximumValue(labelMap)) / 2;
  iftImage *res = iftForceLabelMapConnectivity(labelMap, minSize); 
  printf("Post-processing %d to %d regions\n", iftMaximumValue(labelMap), iftMaximumValue(res));

  // Clean-up
  free(labelArr);
  iftDestroyImage(&labelMap);
  free(minDist);

  return res;
}

void UpdateClusterCentersSp(iftSuperpixelIGraph *igraph, iftMatrix *labelMx)
{
  // Reset existing seed data
  for (int i = 0; i < igraph->seedFeats->n; ++i)
    igraph->seedFeats->val[i] = 0.0f;
  for (int i = 0; i < igraph->seedPos->n; ++i)
    igraph->seedPos->val[i] = 0.0f;

  // Init
  int nClusters = igraph->seedFeats->nrows; // shorthand
  int *clusterSize = iftAllocIntArray(nClusters);
  int *clusterSizePx = iftAllocIntArray(nClusters);
  int **indexedRegionSize = (int **) malloc(nClusters * sizeof(*indexedRegionSize));

  // Find cluster sizes in regions from previous level and pixel count
  for (int node = 0; node < igraph->nNodes; ++node) {
    for (int c = 0; c < nClusters; ++c) {
      if (iftMatrixElem(labelMx, c, node) > 0.0f) {
        clusterSize[c] += 1;
        clusterSizePx[c] += igraph->spSize[node]; 
      }
    }
  }

  // Find previous level regions size in px indexed by their cluster
  for (int c = 0; c < nClusters; ++c)
    indexedRegionSize[c] = iftAllocIntArray(clusterSize[c]);
  int *auxIdx = iftAllocIntArray(nClusters);
  for (int node = 0; node < igraph->nNodes; ++node)
    for (int c = 0; c < nClusters; ++c)
      if (iftMatrixElem(labelMx, c, node) > 0.0f)
        indexedRegionSize[c][(auxIdx[c])++] += igraph->spSize[node];

  // Prepare to sort each feature in each cluster
  float ***aux = (float ***) malloc(nClusters * sizeof(*aux));
  int *clusterIndex = iftAllocIntArray(nClusters);
  for (int c = 0; c < nClusters; ++c) {
    aux[c] = (float **) malloc(igraph->feats->ncols * sizeof(**aux));
    for (int f = 0; f < igraph->feats->ncols; ++f)
      aux[c][f] = iftAllocFloatArray(clusterSize[c]);
  }

  // Accumulation 
  for (int node = 0; node < igraph->nNodes; ++node) {
    for (int c = 0; c < nClusters; ++c) {
      if (iftMatrixElem(labelMx, c, node) > 0.0f) {
        // For features the region size only appears during sorting
        for (int f = 0; f < igraph->feats->ncols; ++f)
          aux[c][f][clusterIndex[c]] = 
            iftMatrixRowPointer(igraph->feats, node)[f]; 
        // For centroid the region size acts as a weighted mean
        for (int f = 0; f < igraph->pos->ncols; ++f) {
          iftMatrixRowPointer(igraph->seedPos, c)[f] += 
            iftMatrixRowPointer(igraph->pos, node)[f] * igraph->spSize[node];
        } 
        clusterIndex[c] += 1;
      }
    }
  }

  // Weighted mean for centroid value
  for (int c = 0; c < nClusters; ++c)
    for (int f = 0; f < igraph->pos->ncols; ++f)
      iftMatrixRowPointer(igraph->seedPos, c)[f] /= ((float) clusterSizePx[c]);

  // Weighted median for parametric value
  for (int c = 0; c < nClusters; ++c) {
    for (int f = 0; f < igraph->feats->ncols; ++f) {
      // Sort and select weighted median
      iftFHeapSort(aux[c][f], indexedRegionSize[c], clusterSize[c], IFT_INCREASING);
      float median = IFT_INFINITY_FLT;
      float pxMedian = clusterSizePx[c] / 2;
      int tracker = 0;
      for (int i = 0; i < clusterSize[c]; ++i) {
        tracker += indexedRegionSize[c][i];
        if (tracker > pxMedian) {
          median = aux[c][f][i];
          break;
        }
      }
      iftMatrixRowPointer(igraph->seedFeats, c)[f] = median;
    }
  }

  // Clean up
  free(clusterSize);
  for (int c = 0; c < nClusters; ++c) {
    for (int f = 0; f < igraph->feats->ncols; ++f)
      free(aux[c][f]);
    free(aux[c]);
    free(indexedRegionSize[c]);
  }
  free(aux);
  free(indexedRegionSize);
  free(clusterIndex);
}

void UpdateClusterCentersPx(iftSuperpixelIGraph *igraph, int *labelArr)
{
  // Reset existing seed data
  for (int i = 0; i < igraph->seedFeats->n; ++i)
    igraph->seedFeats->val[i] = 0.0f;
  for (int i = 0; i < igraph->seedPos->n; ++i)
    igraph->seedPos->val[i] = 0.0f;

  // Init
  int nClusters = igraph->seedFeats->nrows; // shorthand
  int *clusterSize = iftAllocIntArray(nClusters);

  // Find cluster sizes
  for (int node = 0; node < igraph->nNodes; ++node)
    for (int c = 0; c < nClusters; ++c)
      if (labelArr[node] == c)
        clusterSize[c] += 1;

  // Prepare to sort each feature in each cluster
  float ***aux = (float ***) malloc(nClusters * sizeof(*aux));
  int *clusterIndex = iftAllocIntArray(nClusters);
  for (int c = 0; c < nClusters; ++c) {
    aux[c] = (float **) malloc(igraph->feats->ncols * sizeof(**aux));
    for (int f = 0; f < igraph->feats->ncols; ++f)
      aux[c][f] = iftAllocFloatArray(clusterSize[c]);
  }

  // Accumulation 
  for (int node = 0; node < igraph->nNodes; ++node) {
    for (int c = 0; c < nClusters; ++c) {
      if (labelArr[node] == c) {
        // Medoid
        for (int f = 0; f < igraph->feats->ncols; ++f)
          aux[c][f][clusterIndex[c]] = 
            iftMatrixRowPointer(igraph->feats, node)[f]; 
        // Centroid
        for (int f = 0; f < igraph->pos->ncols; ++f) {
          iftMatrixRowPointer(igraph->seedPos, c)[f] += 
            iftMatrixRowPointer(igraph->pos, node)[f];
        } 
        clusterIndex[c] += 1;
      }
    }
  }

  // Final weighted mean for centroid value
  for (int c = 0; c < nClusters; ++c)
    for (int f = 0; f < igraph->pos->ncols; ++f)
      iftMatrixRowPointer(igraph->seedPos, c)[f] /= ((float) clusterSize[c]);

  // Final median for parametric value
  for (int c = 0; c < nClusters; ++c) {
    int *idxDummy = iftAllocIntArray(clusterSize[c]);
    for (int f = 0; f < igraph->feats->ncols; ++f) {
      // Sort and select weighted median
      iftFHeapSort(aux[c][f], idxDummy, clusterSize[c], IFT_INCREASING);
      float median = aux[c][f][clusterSize[c] / 2];
      iftMatrixRowPointer(igraph->seedFeats, c)[f] = median;
    }
    free(idxDummy);
  }

  // Clean up
  free(clusterSize);
  for (int c = 0; c < nClusters; ++c) {
    for (int f = 0; f < igraph->feats->ncols; ++f)
      free(aux[c][f]);
    free(aux[c]);
  }
  free(aux);
  free(clusterIndex);
}

void UpdateClusterCentersSpNoMerge(iftSuperpixelIGraph *igraph, int *labelArr)
{
  // Reset existing seed data
  for (int i = 0; i < igraph->seedFeats->n; ++i)
    igraph->seedFeats->val[i] = 0.0f;
  for (int i = 0; i < igraph->seedPos->n; ++i)
    igraph->seedPos->val[i] = 0.0f;

  // Init
  int nClusters = igraph->seedFeats->nrows; // shorthand
  int *clusterSize = iftAllocIntArray(nClusters);

  // Find cluster sizes
  for (int node = 0; node < igraph->nNodes; ++node)
    for (int c = 0; c < nClusters; ++c)
      if (labelArr[node] == c)
        clusterSize[c] += igraph->spSize[node];

  // Accumulation 
  for (int node = 0; node < igraph->nNodes; ++node) {
    for (int c = 0; c < nClusters; ++c) {
      if (labelArr[node] == c) {
        // Medoid
        for (int f = 0; f < igraph->feats->ncols; ++f)
          iftMatrixRowPointer(igraph->seedFeats, c)[f] += 
            iftMatrixRowPointer(igraph->feats, node)[f] * igraph->spSize[node]; 
        // Centroid
        for (int f = 0; f < igraph->pos->ncols; ++f)
          iftMatrixRowPointer(igraph->seedPos, c)[f] += 
            iftMatrixRowPointer(igraph->pos, node)[f] * igraph->spSize[node];
      }
    }
  }

  // Weighted mean
  for (int c = 0; c < nClusters; ++c) {
    for (int f = 0; f < igraph->feats->ncols; ++f)
      iftMatrixRowPointer(igraph->seedFeats, c)[f] /= ((float) clusterSize[c]);
    for (int f = 0; f < igraph->pos->ncols; ++f)
      iftMatrixRowPointer(igraph->seedPos, c)[f] /= ((float) clusterSize[c]);
  }

  // Clean up
  free(clusterSize);
}

void MergeOverlappingClusters(iftSuperpixelIGraph *igraph, iftMatrix **labelMx, int *pNClusters)
{
  int nClusters = *pNClusters;
  iftMatrix *prevMx = *labelMx;

  // Union-find based algorithm
  // Init
  int *parent = iftAllocIntArray(nClusters);
  int *rank = iftAllocIntArray(nClusters);
  for (int c = 0; c < nClusters; ++c) {
    parent[c] = c;
    rank[c] = 1;
  }

  // Find nodes with overlapping clusters
  for (int c1 = 0; c1 < nClusters; ++c1) {
    for (int c2 = c1 + 1; c2 < nClusters; ++c2) {
      for (int node = 0; node < igraph->nNodes; ++node) {
        if (iftMatrixElem(prevMx, c1, node) < 0.0f)
          continue;
        if (iftMatrixElem(prevMx, c2, node) < 0.0f)
          continue;
        set_union(parent, rank, c1, c2);
      }
    }
  }

  // Count unique clusters and create mapping onto reduced space
  int uniqueCount = 0;
  int *mapToUnique = iftAllocIntArray(nClusters);
  int uniqueLabelIdx = 0;
  for (int c = 0; c < nClusters; ++c) {
    if (c == parent[c]) {
      uniqueCount += 1;
      mapToUnique[c] = uniqueLabelIdx++;
    }
  }

  // Check if no merge occurred
  if (uniqueCount == nClusters) {
    free(parent);
    free(rank);
    free(mapToUnique);
    return;
  }

  printf("Collapsing from %d to %d clusters.\n", nClusters, uniqueCount);

  // Map nodes to final cluster label
  int *nodeMap = iftAllocIntArray(igraph->nNodes);
  for (int node = 0; node < igraph->nNodes; ++node) {
    for (int c = 0; c < nClusters; ++c) {
      if (iftMatrixElem(prevMx, c, node) > 0.0f) {
        nodeMap[node] = mapToUnique[set_find(parent, c)];
        break;
      }
    }
  }

  // Update labelMx accordingly
  iftMatrix *newMx = iftCreateMatrix(uniqueCount, igraph->nNodes); 
  for (int i = 0; i < newMx->n; ++i)
    newMx->val[i] = -1.0f;
  for (int node = 0; node < igraph->nNodes; ++node)
    iftMatrixElem(newMx, nodeMap[node], node) = 1.0f;
  *labelMx = newMx;
  *pNClusters = uniqueCount;

  iftDestroyMatrix(&prevMx);
  free(parent);
  free(rank);
  free(mapToUnique);
  free(nodeMap);
}

int set_find(int *parent, int node)
{
  if (parent[node] != node)
    parent[node] = set_find(parent, parent[node]); 
  return parent[node];
}

void set_union(int *parent, int *rank, int a, int b)
{
  a = set_find(parent, a);
  b = set_find(parent, b);
  if (a == b)
    return;

  if (rank[a] > rank[b]) {
    parent[b] = a;
  } else {
    parent[a] = b;
    if (rank[a] == rank[b])
      rank[b] += 1;
  }
}

void nkComputeNodeToPrototypeDist(iftSuperpixelIGraph *igraph, iftMatrix *featDistToPrototype, iftMatrix *spaceDistToPrototype, int node, int c, float normalizationFactor)
{
  iftMatrixElem(featDistToPrototype, c, node) = igraph->distFun(
      iftMatrixRowPointer(igraph->feats, node),
      iftMatrixRowPointer(igraph->seedFeats, c),
      igraph->distAlpha,
      igraph->feats->ncols);
  iftMatrixElem(spaceDistToPrototype, c, node) = iftFeatDistance(
      iftMatrixRowPointer(igraph->pos, node),
      iftMatrixRowPointer(igraph->seedPos, c),
      igraph->pos->ncols) * normalizationFactor;
}

void nkInitClusters(iftSuperpixelIGraph *igraph, iftImage *initPartition, iftMatrix *labelMx, int nClusters) 
{
  if (initPartition != NULL) {
    for (int p = 0; p < initPartition->n; ++p) {
      int label = initPartition->val[p] - 1;
      int node = igraph->refImg->val[p] - 1;
      if (label < 0 || node < 0)
        continue;
      iftMatrixElem(labelMx, label, node) = 1.0f;
    }
    // Obtain initial prototypes
    igraph->seedFeats = iftCreateMatrix(igraph->feats->ncols, nClusters);
    igraph->seedPos = iftCreateMatrix(igraph->pos->ncols, nClusters);
    UpdateClusterCentersSp(igraph, labelMx);
  } else {
    // Otherwise each node starts as its own cluster
    igraph->seedFeats = iftCopyMatrix(igraph->feats);
    igraph->seedPos = iftCopyMatrix(igraph->pos);
  }
}

extern iftSet * iftRISFGridSampling(  iftSuperpixelIGraph *igraph,   iftImage *img, int nSamples);

iftImage *iftBuildGridInitPartition(iftSuperpixelIGraph *igraph, int nSp)
{
  iftSet *seeds = iftRISFGridSampling(igraph, igraph->refImg, nSp); 
  iftImage *res = iftCreateImage(igraph->refImg->xsize, igraph->refImg->ysize, igraph->refImg->zsize);
  int label = 1;
  while (seeds != NULL) {
    int node = iftRemoveSet(&seeds);
    for (int p = 0; p < res->n; ++p)
      if (igraph->refImg->val[p] - 1 == node)
        res->val[p] = label;
    label++;
  }

  return res;
}

iftSet * nkGetPartitionCenters(iftImage *part)
{
  iftSet *res = NULL;
  iftMatrix *mx = iftComputeSuperpixelFeaturesByGeometricCenter(part);
  for (int row = 0; row < mx->nrows; ++row) {
    iftVoxel u = {
      .x = iftMatrixRowPointer(mx, row)[0], 
      .y = iftMatrixRowPointer(mx, row)[1], 
      .z = 0
    };
    int px = iftGetVoxelIndex(part, u);
    iftInsertSet(&res, px);
  }

  return res;
}
