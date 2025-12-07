/**
 * @file
 * @brief Implementation of VCells superpixel segmentation.
 * @author Felipe Lemes Galvao
 *
 * TODO Include proper paper reference
 */

#include <ift.h>
#include <metis.h>
#include "iftSuperpixelFeatures.h"
#include "iftExperimentUtility.h"

#define INPUT_ARG "--input-image"
#define OUTPUT_ARG "--output-image"
#define NSP_ARG "--superpixel-num"
#define THETA_ARG "--theta"
#define OMEGA_ARG "--omega"
#define ITER_ARG "--iters-num"
#define PART_ARG "--init-partition"
#define FEAT_ARG "--superpixel-feat"
#define PRINT_ARG "--print-opt"
#define PREVSEG_ARG "--prev-seg"
#define MASK_ARG "--mask"
#define FORMAT_ARG "--output-format"

// Meant to be a single linked list but we define it this way to re-use iftSet
typedef struct vc_border_set {
  iftSet *nodes;
  iftSet *neighLabel;
} vcBorderSet;

iftDict *iftGetArguments(int argc, const char *argv[]);
void vcGetBorderSet(iftSuperpixelIGraph *igraph, int *label, vcBorderSet *B);
iftImage * iftSuperpixelSegmentationByVCell(iftSuperpixelIGraph *igraph, iftImage *initPartition, int maxIters);

/* 
 * --- vsGetLocalDisagremeent
 *   Regularity energy term based on explicitly counting local
 *     disagreement with proposed label.
 *
 * Details:
 *   Let n_cluster be the number of pixels in Adj(node) that
 *     AGREE with label cluster. Analogously, ntil_cluster
 *     is the number that DISAGREE with label cluster.
 *   For superpixels, adj(node) is NOT the regular RAG adjacency, but defined
 *     from region to pixels using the pixel adjacency projected onto the
 *     pixels in the border of the region delimited by cluster.
 *   note: zhou et al (2015) accidentally invert the definition of n_cluster
 *
 *   In this implementation, for superpixels we pre-compute the number of pixels
 *     in adj(node) that fall into the RAG neighboring region as their label are
 *     identical regardless. That count is stored in the nClusters x nClusters
 *     matrix Mx. See vsPreComputeSuperpixelNeighborhood.
 */
float vsGetLocalDisagreement(iftSuperpixelIGraph *igraph, int node, int candidateLabel, int *nodeLabel, iftMatrix *Mx);

// If row != col, mx(row, col) is the number of pixels from region col that fall
//   into the neighborhood of region row. If row = col, it is neighboord size of
//   region row. Do note that the matrix is not symmetric in general.
iftMatrix *vsPreComputeSuperpixelNeighborhood(iftSuperpixelIGraph *igraph, bool forceSymmetry);

void vcInitGenerators(iftSuperpixelIGraph *igraph, int *nodeLabel, int nClusters, iftMatrix **pClusterSum, int **pClusterSize);

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
  float omega = iftDictContainKey(OMEGA_ARG, args, NULL) ?
    iftGetDblValFromDict(OMEGA_ARG, args) : 2.0f;
  iftAdjRel *A = is3D ? iftSpheric(omega) : iftCircular(omega);
  // This is not a graph-based method, but still uses the idea of adjacency
  if (isHierarchic)
    iftSetSuperpixelIGraphExplicitRAG(igraph, A);
  else
    iftSetSuperpixelIGraphImplicitAdjacency(igraph, A);

  int nIters = (iftDictContainKey(ITER_ARG, args, NULL)) ?
    iftGetLongValFromDict(ITER_ARG, args) : 30;

  // Set superpixel parametric features
  float alpha = (iftDictContainKey(THETA_ARG, args, NULL)) ?
    iftGetDblValFromDict(THETA_ARG, args) : isHierarchic ?
    1000 : 1.5;
  float beta = 0.0f; // Unused for now
  int featOpt = (iftDictContainKey(FEAT_ARG, args, NULL)) ?
    iftGetLongValFromDict(FEAT_ARG, args) : 1;
  iftMatrix *pFeats = NULL;
  switch (featOpt) {
    case 1: // Color mean
    default:
      {
        if (iftIsColorImage(img))
          pFeats = iftComputeSuperpixelFeaturesByColorSpaceMean(igraph->refImg, img, LABNorm_CSPACE);  
        else
          pFeats = iftComputeSuperpixelFeaturesByColorSpaceMean(igraph->refImg, img, GRAYNorm_CSPACE);  
        iftSetSuperpixelIGraphFeatures(igraph, pFeats, iftDistance1, alpha, beta);
      }
      break;
  }

  // Get initial partition
  iftImage *initPartition = NULL;
  if (iftDictContainKey(PART_ARG, args, NULL)) {
    tmpPath = iftGetStrValFromDict(PART_ARG, args);
    initPartition = iftReadImageByExt(tmpPath);
    free(tmpPath);
  } else if (iftDictContainKey(NSP_ARG, args, NULL) && isHierarchic) {
    int nSp = iftGetLongValFromDict(NSP_ARG, args);
    initPartition = iftBuildMetisInitPartition(igraph, nSp);
  } else {
    printf("Either -s or (-n + -l) must be supplied.\n");
    return 1;
  }

  // Compute segmentation
  timer *t1 = iftTic();
  iftImage *superpixelLabelMap = iftSuperpixelSegmentationByVCell(igraph, initPartition, nIters);
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
      .help = "Lower is more compact. Default = 1.5/1000 for pixels/superpixels."},
    {.short_name = "-b", .long_name = OMEGA_ARG, .has_arg = true,
      .arg_type = IFT_DBL_TYPE, .required = false,
      .help = "Radius of the neighborhoods in the algorithm. Default = 2."},
    {.short_name = "-t", .long_name = ITER_ARG, .has_arg = true,
      .arg_type = IFT_LONG_TYPE, .required = false,
      .help = "Max number of iterations. Default = 30."},
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
    "Implementation of VCell segmentation and its hierarchical version.";

  // Parser Setup
  iftCmdLineParser *parser =
    iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
  iftDict *args = iftParseCmdLine(argc, argv, parser);
  iftDestroyCmdLineParser(&parser);

  return args;
}

iftImage * iftSuperpixelSegmentationByVCell(iftSuperpixelIGraph *igraph, iftImage *initPartition, int maxIters)
{
  assert(igraph != NULL);
  assert(initPartition != NULL);

  // Parameter translation
  float theta = igraph->alpha;

  // Data
  int nClusters = iftMaximumValue(initPartition);
  int *nodeLabel = iftAllocIntArray(igraph->nNodes);
  double cvtEnergy = 0.0f;
  double edgeEnergy = 0.0f;

  // Cluster initialization
  for (int p = 0; p < initPartition->n; ++p) {
    int label = initPartition->val[p] - 1;
    int node = igraph->refImg->val[p] - 1;
    if (label < 0 || node < 0)
      continue;
    if (nodeLabel[node] != 0 && nodeLabel[node] != label)
      iftError("iftSuperpixelSegmentationByVCell", "Initial partition is not a\
          proper grouping of regions.");
    nodeLabel[node] = label;
  }

  /* note for superpixels: the distance metric and energy updates are only
   *   an approximation if mx is not symmetric.
   *  
   * The issue is not addressed in Zhou et al (2015) as their set based
   *   definition of superpixel boundary implies a given pixel is not
   *   counted more than once in the superpixel neighborhood, but that
   *   is necessary to guarantee that mx is symmetric. */
  iftMatrix *mx = NULL;
  if (igraph->type == EXPLICIT)
    mx = vsPreComputeSuperpixelNeighborhood(igraph, false);

  iftMatrix *clusterSum;
  int *clusterSize;
  vcInitGenerators(igraph, nodeLabel, nClusters, &clusterSum, &clusterSize);

  // Compute initial energy
  for (int node = 0; node < igraph->nNodes; ++node) {
    cvtEnergy += igraph->distFun(
        iftMatrixRowPointer(igraph->feats, node),
        iftMatrixRowPointer(igraph->seedFeats, nodeLabel[node]),
        igraph->distAlpha,
        igraph->feats->ncols) * igraph->spSize[node];
    edgeEnergy += vsGetLocalDisagreement(igraph, node, nodeLabel[node], nodeLabel, mx);
  }

  printf("Starting with E_cvt = %f and E_edge = %f\n", cvtEnergy, edgeEnergy);

  // -- Main loop
  for (int iter = 0; iter < maxIters; ++iter) {
    bool hasChanged = false;

    // Build border set
    // Both lists are meant to be used as a single linked list 
    vcBorderSet B = {NULL, NULL};
    vcGetBorderSet(igraph, nodeLabel, &B);

    // Compute distances from border nodes to prototypes
    float lambda = cvtEnergy / (theta * edgeEnergy);
    int lastActiveNode = -1;
    float minDist = IFT_INFINITY_FLT;
    float newFeatDist = IFT_INFINITY_FLT;
    float newNtil = IFT_INFINITY_FLT;
    int newLabel = -1;
    while (B.nodes != NULL) {
      int node = iftRemoveSet(&(B.nodes));
      int cluster = iftRemoveSet(&(B.neighLabel));

      // Detect the start of a new border node as
      //   each node has multiple elements in the list.
      if (lastActiveNode != node)
        minDist = IFT_INFINITY_FLT;

      // Distance to prototype (generator in cvt context)
      float featDist = igraph->distFun(
          iftMatrixRowPointer(igraph->feats, node),
          iftMatrixRowPointer(igraph->seedFeats, cluster),
          igraph->distAlpha,
          igraph->feats->ncols) * igraph->spSize[node];
      float ntil = vsGetLocalDisagreement(igraph, node, cluster, nodeLabel, mx);
      float spaceDist = 2.0f * lambda * ntil;
      // No need to take sqrt as we only want to find minimum
      float dist = featDist + spaceDist;
      if (dist < minDist) {
        minDist = dist;
        newFeatDist = featDist;
        newNtil = ntil;
        newLabel = cluster;
      }

      // Detect last element for a given node
      if (B.nodes == NULL || B.nodes->elem != node) {
        // Update energy & label if new assignment is found
        if (newLabel != nodeLabel[node]) {
          int prevLabel = nodeLabel[node];
          nodeLabel[node] = newLabel;
          hasChanged = true;

          // Update cvt energy
          float prevFeatDist = igraph->distFun(
                iftMatrixRowPointer(igraph->feats, node),
                iftMatrixRowPointer(igraph->seedFeats, prevLabel),
                igraph->distAlpha,
                igraph->feats->ncols) * igraph->spSize[node];
          cvtEnergy += (newFeatDist - prevFeatDist); 

          // Update edge energy
          // 2x comes from the fact it changes the neighbors' energies as well
          float prevNtil = vsGetLocalDisagreement(igraph, node, prevLabel, nodeLabel, mx);
          edgeEnergy += (2.0f * (newNtil - prevNtil));

          // Update centroids
          clusterSize[prevLabel] -= igraph->spSize[node];
          clusterSize[newLabel] += igraph->spSize[node];
          for (int f = 0; f < igraph->feats->ncols; ++f) {
            float sumChange = iftMatrixRowPointer(igraph->feats, node)[f]; 
            iftMatrixRowPointer(clusterSum, prevLabel)[f] -= sumChange;
            iftMatrixRowPointer(igraph->seedFeats, prevLabel)[f] =
              iftMatrixRowPointer(clusterSum, prevLabel)[f] 
              / clusterSize[prevLabel];
            iftMatrixRowPointer(clusterSum, newLabel)[f] += sumChange;
            iftMatrixRowPointer(igraph->seedFeats, newLabel)[f] =
              iftMatrixRowPointer(clusterSum, newLabel)[f] 
              / clusterSize[newLabel];
          }
        }
      }

      lastActiveNode = node;
    }

    // Check for convergence
    if (!hasChanged) {
      printf("Converged early at iter %d\n", iter+1);
      break;
    }
  }

  printf("Ended with E_cvt = %f and E_edge = %f\n", cvtEnergy, edgeEnergy);

  // Build initial label map
  iftImage *labelMap = iftCopyImage(igraph->refImg);
  for (int p = 0; p < labelMap->n; ++p) {
    int node = labelMap->val[p] - 1;
    if (node < 0)
      continue;
    labelMap->val[p] = nodeLabel[node]; 
  }

  // SLIC-like post-processing to fix unconnected regions & remove small ones
  int minSize = igraph->type == IMPLICIT ? ((initPartition->n / nClusters) / 2): 0;
  iftImage *res = iftForceLabelMapConnectivity(labelMap, minSize); 

  free(nodeLabel);
  iftDestroyMatrix(&mx);
  iftDestroyMatrix(&clusterSum);
  free(clusterSize);
  iftDestroyImage(&labelMap);

  return res;
}

void vcInitGenerators(iftSuperpixelIGraph *igraph, int *nodeLabel, int nClusters, iftMatrix **pClusterSum, int **pClusterSize)
{
  // Allocation
  int *clusterSize = iftAllocIntArray(nClusters);
  igraph->seedFeats = iftCreateMatrix(igraph->feats->ncols, nClusters);

  // Weighted mean accumulation
  for (int node = 0; node < igraph->nNodes; ++node) {
    int cluster = nodeLabel[node];
    for (int f = 0; f < igraph->feats->ncols; ++f)
      iftMatrixRowPointer(igraph->seedFeats, cluster)[f] += 
        iftMatrixRowPointer(igraph->feats, node)[f] * igraph->spSize[node]; 
    clusterSize[cluster] += igraph->spSize[node];
  }

  // Store secondary results
  *pClusterSum = iftCopyMatrix(igraph->seedFeats);
  *pClusterSize = clusterSize;

  // Divide by total weight to get proper centroid
  for (int c = 0; c < nClusters; ++c)
    for (int f = 0; f < igraph->seedFeats->ncols; ++f)
      iftMatrixRowPointer(igraph->seedFeats, c)[f] /= ((float) clusterSize[c]);
}

void vcGetBorderSet(iftSuperpixelIGraph *igraph, int *label, vcBorderSet *B)
{
  iftSet *visited = NULL;
  if (igraph->type == EXPLICIT) {
    for (int node = 0; node < igraph->nNodes; ++node) {
      for (iftSet *s = igraph->nodeAdj[node]; s != NULL; s = s->next) {
        int neighLabel = label[s->elem];
        if (label[node] != neighLabel && !iftSetHasElement(visited, neighLabel)) {
          iftInsertSet(&visited, neighLabel);
          iftInsertSet(&(B->nodes), node);
          iftInsertSet(&(B->neighLabel), label[s->elem]);
        }
      }
      if (visited != NULL) {
        iftInsertSet(&(B->nodes), node);
        iftInsertSet(&(B->neighLabel), label[node]);
        iftDestroySet(&visited);
      }
    }
  } else {
    // TODO support masks (refer to refImg index instead of p and q)
    for (int p = 0; p < igraph->refImg->n; ++p) {
      iftVoxel u = iftGetVoxelCoord(igraph->refImg, p);
      for (int i = 1; i < igraph->A->n; ++i) {
        iftVoxel v = iftGetAdjacentVoxel(igraph->A, u, i);
        if (!iftValidVoxel(igraph->refImg, v))
          continue;
        int q = iftGetVoxelIndex(igraph->refImg, v); 
        if (label[p] != label[q] && !iftSetHasElement(visited, label[q])) { 
          iftInsertSet(&visited, label[q]);
          iftInsertSet(&(B->nodes), p);
          iftInsertSet(&(B->neighLabel), label[q]);
        }
      }
      if (visited != NULL) {
        iftInsertSet(&(B->nodes), p);
        iftInsertSet(&(B->neighLabel), label[p]);
        iftDestroySet(&visited);
      }
    }
  }
}

iftMatrix *vsPreComputeSuperpixelNeighborhood(iftSuperpixelIGraph *igraph, bool forceSymmetry)
{
  // Pre-compute neighboring pixel count with each particular cluster label
  iftMatrix *mx = iftCreateMatrix(igraph->nNodes, igraph->nNodes);
  iftAdjRel *Aw = igraph->A;
  // Hash would be assymptotically better but in practice the set should be small
  //   with any reasonable Aw radius.
  iftSet *visitedRegions = NULL;

  for (int p = 0; p < igraph->refImg->n; ++p) {
    int pLabel = igraph->refImg->val[p] - 1;
    if (pLabel < 0)
      continue;
    iftVoxel u = iftGetVoxelCoord(igraph->refImg, p);
    for (int i = 1; i < Aw->n; ++i) {
      iftVoxel v = iftGetAdjacentVoxel(Aw, u, i);
      if (!iftValidVoxel(igraph->refImg, v))
        continue;
      int q = iftGetVoxelIndex(igraph->refImg, v);
      int qLabel = igraph->refImg->val[q] - 1;
      if (qLabel < 0)
        continue;
      if ((pLabel != qLabel) && !iftSetHasElement(visitedRegions, qLabel)) {
        if (!forceSymmetry)
          iftInsertSet(&visitedRegions, qLabel);
        iftMatrixElem(mx, qLabel, pLabel) += 1.0f;
        iftMatrixElem(mx, pLabel, pLabel) += 1.0f;
      }
    }
    iftDestroySet(&visitedRegions);
  }

  return mx;
}

// Result is technically an integer (we are counting pixels) 
float vsGetLocalDisagreement(iftSuperpixelIGraph *igraph, int node, int candidateLabel, int *nodeLabel, iftMatrix *Mx)
{
  float res = 0.0f;
  if (igraph->type == EXPLICIT) {
    assert(Mx != NULL);
    for (iftSet *s = igraph->nodeAdj[node]; s != NULL; s = s->next) {
      if (candidateLabel != nodeLabel[s->elem]) {
        res += iftMatrixElem(Mx, s->elem, node);
      }
    }
  } else {
    iftVoxel u = iftGetVoxelCoord(igraph->refImg, node);
    for (int i = 1; i < igraph->A->n; ++i) {
      iftVoxel v = iftGetAdjacentVoxel(igraph->A, u, i);
      // We assume voxels outside the image always agree
      if (!iftValidVoxel(igraph->refImg, v))
        continue;
      int neigh = iftGetVoxelIndex(igraph->refImg, v);
      if (candidateLabel != nodeLabel[neigh])
        res += 1.0f;
    }
  }

  return res;
}

