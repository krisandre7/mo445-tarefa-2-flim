/**
 * @file
 * @brief Segmentation by Felzenszwalb & Huttenlocher method.
 * @author Felipe Lemes Galvao
 *
 *   See Felzenszwalb, P.F. & Huttenlocher, D.P. IJCV (2004).
 *   We implement a more general version that works over RAGs and
 *     different image features.
 */

#include <ift.h>
#include "iftSuperpixelFeatures.h"
#include "iftExperimentUtility.h"

#define INPUT_ARG "--input-image"
#define OUTPUT_ARG "--output-image"
#define SCALE_ARG "--k-scale"
#define MINSIZE_ARG "--min-size"
#define FEAT_ARG "--feat"
#define PRINT_ARG "--print-opt"
#define PREVSEG_ARG "--prev-seg"
#define MASK_ARG "--mask"
#define FORMAT_ARG "--output-format"

extern void iftRemoveLabelGaps(int *labelArray, int size);

iftDict *iftGetArguments(int argc, const char *argv[]);
iftImage * iftSuperpixelSegmentationByFH(iftSuperpixelIGraph *igraph, iftImage *img, int minSize);
int fh_find(int *parent, int node);
void fh_union(int *parent, int *rank, int a, int b);

typedef struct graph_edge {
  int a;
  int b;
} GraphEdge;

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

  // Prepare igraph reference image
  iftImage *refImg = NULL;
  if (isHierarchic) {
    // Load pre-computed superpixel segmentation (finer level)
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
      refImg = iftCreateImage(img->xsize, img->ysize, img->zsize);
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
  iftAdjRel *A = is3D ? iftSpheric(sqrt(3.0)) : iftCircular(sqrt(2.0));
  if (isHierarchic)
    iftSetSuperpixelIGraphExplicitRAG(igraph, A);
  else
    iftSetSuperpixelIGraphImplicitAdjacency(igraph, A);

  // Set superpixel parametric features
  int minSize = iftGetLongValFromDict(MINSIZE_ARG, args);
  float alpha = iftGetDblValFromDict(SCALE_ARG, args) / 255.0;
  float beta = 1.0; // Dummy
  int featOpt = (iftDictContainKey(FEAT_ARG, args, NULL)) ?
    iftGetLongValFromDict(FEAT_ARG, args) : 1;
  iftMatrix *pFeats = NULL;
  switch (featOpt) {
    case 1: // LAB mean
    default:
      if (iftIsColorImage(img))
        pFeats = iftComputeSuperpixelFeaturesByColorSpaceMean(igraph->refImg, img, RGBNorm_CSPACE);  
      else
        pFeats = iftComputeSuperpixelFeaturesByColorSpaceMean(igraph->refImg, img, GRAYNorm_CSPACE);  
      double maxFeat = 0.0;
      for (int i = 0; i < pFeats->n; ++i)
        if (pFeats->val[i] > maxFeat)
          maxFeat = pFeats->val[i];
      iftSetSuperpixelIGraphFeatures(igraph, pFeats, iftDistance1, alpha, beta);
      break;
    case 2: // 20^3-bins LAB histogram XiSquare dist
      pFeats = iftComputeSuperpixelFeaturesByColorHistogram(img, igraph->refImg,  20);
      for (int node = 0; node < pFeats->nrows; ++node)
        iftNormalizeFeatures(iftMatrixRowPointer(pFeats, node), pFeats->ncols);
      iftSetSuperpixelIGraphFeatures(igraph, pFeats, iftDistance10, alpha, beta);
      break;
    case 3: // RGB 
      printf("Implement pls\n");
      return -1;
  }

  // Compute segmentation
  timer *t1 = iftTic();
  iftImage *superpixelLabelMap = iftSuperpixelSegmentationByFH(igraph, img, minSize);
  timer *t2 = iftToc();
  iftRemoveLabelGaps(superpixelLabelMap->val,
      superpixelLabelMap->n);
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

  iftDestroyDict(&args);
  iftDestroyImage(&img);
  iftDestroyImage(&refImg);
  iftDestroySuperpixelIGraph(&igraph);
  iftDestroyAdjRel(&A);
  iftDestroyMatrix(&pFeats);
  iftDestroyImage(&superpixelLabelMap);

  return 0;
}

#define FEAT_HELP "Superpixel features:\n\
    1 = LAB mean\n\
    2 = LAB histogram\n\
    3 = RGB per channel (DEFAULT)"

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
    {.short_name = "-k", .long_name = SCALE_ARG, .has_arg = true,
      .arg_type = IFT_DBL_TYPE, .required = true,
      .help = "FH k parameter, increase to reduce num of sp."},
    {.short_name = "-s", .long_name = MINSIZE_ARG, .has_arg = true,
      .arg_type = IFT_LONG_TYPE, .required = true,
      .help = "FH mininum superpixel size parameter."},
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
    "This demo runs FH segmentation with some additional options.";

  // Parser Setup
  iftCmdLineParser *parser =
    iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
  iftDict *args = iftParseCmdLine(argc, argv, parser);
  iftDestroyCmdLineParser(&parser);

  return args;
}

iftImage * iftSuperpixelSegmentationByFH(iftSuperpixelIGraph *igraph, iftImage *img, int minSize)
{ 
  // Memory allocation for edge information
  int nEdges = 0;
  assert(igraph->type != COMPLETE);
  if (igraph->type == IMPLICIT)
    nEdges = igraph->nNodes * (igraph->A->n - 1);
  else
    for (int i = 0; i < igraph->nNodes; ++i)
      nEdges += iftSetSize(igraph->nodeAdj[i]);

  float *weight = iftAllocFloatArray(nEdges);
  GraphEdge *edge = malloc(nEdges * (sizeof(*edge)));
  int *index = iftAllocIntArray(nEdges);

  for (int i = 0; i < nEdges; ++i) {
    weight[i] = IFT_INFINITY_FLT; // For unused edges
    index[i] = i;
  }

  // Compute edge weights based on node features
  int edgeIdx = 0;
  if (igraph->type == IMPLICIT) {
    for (int p = 0; p < igraph->refImg->n; ++p) {
      iftVoxel u = iftGetVoxelCoord(igraph->refImg, p);
      for (int i = 1; i < igraph->A->n; ++i) {
        iftVoxel v = iftGetAdjacentVoxel(igraph->A, u, i);
        if (!iftValidVoxel(igraph->refImg, v))
          continue;

        int q = iftGetVoxelIndex(igraph->refImg, v);
        weight[edgeIdx] = igraph->distFun(
            iftMatrixRowPointer(igraph->feats, p),
            iftMatrixRowPointer(igraph->feats, q),
            igraph->distAlpha,
            igraph->feats->ncols);
        edge[edgeIdx].a = p;
        edge[edgeIdx].b = q;
        edgeIdx += 1;
      }
    }
  } else {
    for (int node = 0; node < igraph->nNodes; ++node) {
      for (iftSet *adj = igraph->nodeAdj[node]; adj != NULL; adj = adj->next) {
        assert(edgeIdx < nEdges);
        weight[edgeIdx] = igraph->distFun(
            iftMatrixRowPointer(igraph->feats, node),
            iftMatrixRowPointer(igraph->feats, adj->elem),
            igraph->distAlpha,
            igraph->feats->ncols);
        edge[edgeIdx].a = node;
        edge[edgeIdx].b = adj->elem;
        edgeIdx += 1;
      }
    }
  }

  // Sort edges, should be the most expensive operation
  // Edge array should be accessed through index array afterwards
  iftFHeapSort(weight, index, nEdges, IFT_INCREASING);

  // Use Union-Find algorithm to build segmentation
  int *parent = iftAllocIntArray(igraph->nNodes);
  int *rank = iftAllocIntArray(igraph->nNodes);
  int *size = iftAllocIntArray(igraph->nNodes);
  float *th = iftAllocFloatArray(igraph->nNodes);
  float k = igraph->alpha;
  for (int i = 0; i < igraph->nNodes; ++i) {
    parent[i] = i;
    size[i] = 1;
    th[i] = k;
  }
  // Use underlying region size instead
  /*for (int p = 0; p < igraph->refImg->n; ++p) {
    int label = igraph->refImg->val[i] - 1;
    if (label >= 0)
      size[label] += 1;
  }*/

  for (int e = 0; e < nEdges; ++e) {
    if (weight[e] == IFT_INFINITY_FLT)
      break;

    int a = fh_find(parent, edge[index[e]].a);
    int b = fh_find(parent, edge[index[e]].b);
    if (a == b)
      continue;

    if (weight[e] <= iftMin(th[a], th[b])) {
      fh_union(parent, rank, a, b);
      int c = fh_find(parent, a);
      size[c] = size[a] + size[b];
      th[c] = weight[e] + k / size[c];
    }
  }

  // Post-processing to remove small components
  for (int e = 0; e < nEdges; ++e) {
    int a = fh_find(parent, edge[index[e]].a);
    int b = fh_find(parent, edge[index[e]].b);
    if ((a != b) && ((size[a] < minSize) || (size[b] < minSize))) {
      fh_union(parent, rank, a, b);
      int c = fh_find(parent, a);
      size[c] = size[a] + size[b];
    }
  }

  // Build label map from result
  iftImage *res = iftCopyImage(igraph->refImg);
  for (int p = 0; p < res->n; ++p) {
    int label = igraph->refImg->val[p] - 1;
    if (label < 0)
      continue;

    res->val[p] = fh_find(parent, label);
  }

  // Free memory
  free(weight);
  free(edge);
  free(index);
  free(parent);
  free(rank);
  free(size);
  free(th);

  return res;
}

int fh_find(int *parent, int node)
{
  if (parent[node] != node)
    parent[node] = fh_find(parent, parent[node]); 
  return parent[node];
}

void fh_union(int *parent, int *rank, int a, int b)
{
  // In this context a and b are already roots
  if (rank[a] > rank[b]) {
    parent[b] = a;
  } else {
    parent[a] = b;
    if (rank[a] == rank[b])
      rank[b] += 1;
  }
}
