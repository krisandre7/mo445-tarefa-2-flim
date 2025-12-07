/**
 * @file
 * @brief Segmentation by greedy merge.
 * @author Felipe Lemes Galvao
 */

#include <ift.h>
#include "iftSuperpixelFeatures.h"
#include "iftExperimentUtility.h"

#define INPUT_ARG "--input-image"
#define OUTPUT_ARG "--output-image"
#define NSP_ARG "--num-regions"
#define FEAT_ARG "--feat"
#define PRINT_ARG "--print-opt"
#define PREVSEG_ARG "--prev-seg"
#define MASK_ARG "--mask"
#define FORMAT_ARG "--output-format"

extern void iftRemoveLabelGaps(int *labelArray, int size);

iftDict *iftGetArguments(int argc, const char *argv[]);
int prgFind(int *parent, int node);
void prgUnion(int *parent, int *rank, int a, int b);
iftImage * iftSuperpixelSegmentationByPRG_Heap(iftSuperpixelIGraph *igraph);
float prgArcWeight(iftSuperpixelIGraph *igraph, int i, int j);
void prgMergeNodesInGraph(iftSuperpixelIGraph *igraph, int i, int j);
iftImage * prgConvertToLabelImage(iftSuperpixelIGraph *igraph, int *parent);
float prgMergeEnergyDelta(iftSuperpixelIGraph *igraph, int i, int j, int *parent);
void prgInitSeedMx(iftSuperpixelIGraph *igraph);
void prgNormalizeAdj(iftSet **s, int *parent);
void iftUpdateDHeapElem(iftDHeap *H, int node, double val);

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
    refImg = iftForceLabelMapConnectivity(refImg, 10);
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
  float alpha = 0.0f;
  float beta = 1.0; 
  int featOpt = (iftDictContainKey(FEAT_ARG, args, NULL)) ?
    iftGetLongValFromDict(FEAT_ARG, args) : 1;
  iftMatrix *pFeats = NULL;
  switch (featOpt) {
    case 1: // LAB mean
    default:
      if (iftIsColorImage(img))
        pFeats = iftComputeSuperpixelFeaturesByColorSpaceMean(igraph->refImg, img, LABNorm_CSPACE);  
        //pFeats = iftComputeSuperpixelFeaturesByColorSpaceMean(igraph->refImg, img, RGBNorm_CSPACE);  
      else
        pFeats = iftComputeSuperpixelFeaturesByColorSpaceMean(igraph->refImg, img, GRAYNorm_CSPACE);  
      iftSetSuperpixelIGraphFeatures(igraph, pFeats, iftDistance1, alpha, beta);
      break;
    case 2: // 20^3-bins LAB histogram XiSquare dist
      pFeats = iftComputeSuperpixelFeaturesByColorHistogram(img, igraph->refImg, 20);
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
  iftImage *superpixelLabelMap = iftSuperpixelSegmentationByPRG_Heap(igraph);
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
    {.short_name = "-f", .long_name = FEAT_ARG, .has_arg = true,
      .arg_type = IFT_LONG_TYPE, .required = false,
      .help = FEAT_HELP},
    {.short_name = "-p", .long_name = PRINT_ARG, .has_arg = true,
      .arg_type = IFT_LONG_TYPE, .required = false,
      .help = PRINT_HELP},
    {.short_name = "-l", .long_name = PREVSEG_ARG, .has_arg = true,
      .arg_type = IFT_STR_TYPE, .required = true,
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
    "This demo runs greedy merge segmentation with some additional options.";

  // Parser Setup
  iftCmdLineParser *parser =
    iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
  iftDict *args = iftParseCmdLine(argc, argv, parser);
  iftDestroyCmdLineParser(&parser);

  return args;
}

int prgFind(int *parent, int node)
{
  if (parent[node] != node)
    parent[node] = prgFind(parent, parent[node]); 
  return parent[node];
}

void prgUnion(int *parent, int *rank, int a, int b)
{
  a = prgFind(parent, a);
  b = prgFind(parent, b);

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

// NOTE: igraph itself will be invalid afterwards
iftImage * iftSuperpixelSegmentationByPRG_Heap(iftSuperpixelIGraph *igraph)
{
  // Union-find data init
  int * parent = iftAllocIntArray(igraph->nNodes);
  int * rank = iftAllocIntArray(igraph->nNodes);
  for (int i = 0; i < igraph->nNodes; ++i) {
    parent[i] = i;
    rank[i] = 1;
  }

  // Feats/Pos matrix holds weighted sum
  // seedFeatsPos counterpart is that value divided by size for actual mean
  //   plus an additional field for working data (see merge energy delta)
  prgInitSeedMx(igraph);
  for (int i = 0; i < igraph->nNodes; ++i) {
    for (int f = 0; f < igraph->feats->ncols; ++f)
      iftMatrixRowPointer(igraph->feats, i)[f] *= igraph->spSize[i];
    for (int f = 0; f < igraph->pos->ncols; ++f)
      iftMatrixRowPointer(igraph->pos, i)[f] *= igraph->spSize[i];
  }

  // Start merging loop
  int MIN_OBJ_SIZE = 1000;
  int MAX_OBJ_SIZE = 3000;
  for (int i = 0; i < igraph->nNodes; ++i) {
    int node = prgFind(parent, i);
    if (igraph->spSize[node] < MIN_OBJ_SIZE) {
      // Merge with closest node except if it would push a smaller one above big threshold
      float minDist = IFT_INFINITY_FLT;
      int minNeigh = -1;
      for (iftSet *s = igraph->nodeAdj[node]; s != NULL; s = s->next) {
        int j = s->elem;
        int neigh = prgFind(parent, j);
        if (node == neigh)
          continue;

        if (igraph->spSize[neigh] < MAX_OBJ_SIZE && igraph->spSize[node] + igraph->spSize[neigh] > MAX_OBJ_SIZE)
          continue;

        float edgeVal = prgArcWeight(igraph, node, neigh)
          * iftMin(igraph->spSize[node], igraph->spSize[neigh]);
        if (edgeVal < minDist) {
          minDist = edgeVal;
          minNeigh = neigh;
        }
      }
      if (minNeigh >= 0) {
        prgUnion(parent, rank, node, minNeigh);
        if (node != parent[node])
          iftSwap(node, minNeigh);
        prgMergeNodesInGraph(igraph, node, minNeigh);

        for (iftSet *s = igraph->nodeAdj[node]; s != NULL; s = s->next)
          s->elem = prgFind(parent, s->elem);
        iftSet *auxSet = igraph->nodeAdj[node];
        igraph->nodeAdj[node] = iftSetUnion(NULL, auxSet);
        iftDestroySet(&auxSet);
      }
    } else if (igraph->spSize[node] > MAX_OBJ_SIZE) {
      // Merge with any neighbor above big threshold
      bool mergeHappened;
      do {
        mergeHappened = false;
        for (iftSet *s = igraph->nodeAdj[node]; s != NULL; s = s->next) {
          int j = s->elem;
          int neigh = prgFind(parent, j);
          if (node == neigh)
            continue;
          if (igraph->spSize[neigh] > MAX_OBJ_SIZE) {
            mergeHappened = true;

            prgUnion(parent, rank, node, neigh);
            if (node != parent[node])
              iftSwap(node, neigh);
            prgMergeNodesInGraph(igraph, node, neigh);

            for (iftSet *s = igraph->nodeAdj[node]; s != NULL; s = s->next)
              s->elem = prgFind(parent, s->elem);
            iftSet *auxSet = igraph->nodeAdj[node];
            igraph->nodeAdj[node] = iftSetUnion(NULL, auxSet);
            iftDestroySet(&auxSet);

            break;
          }
        }
      } while(mergeHappened);
    }
  }

  iftImage *res = prgConvertToLabelImage(igraph, parent);

  free(parent);
  free(rank);

  return res;
}

float prgArcWeight(iftSuperpixelIGraph *igraph, int i, int j) 
{
  float featDist = igraph->distFun(
      iftMatrixRowPointer(igraph->seedFeats, i),
      iftMatrixRowPointer(igraph->seedFeats, j),
      igraph->distAlpha,
      igraph->seedFeats->ncols);
  float spaceDist = iftFeatDistance(
      iftMatrixRowPointer(igraph->seedPos, i),
      iftMatrixRowPointer(igraph->seedPos, j),
      igraph->seedPos->ncols);
  
  return (featDist + (igraph->alpha * spaceDist));
}

void prgMergeNodesInGraph(iftSuperpixelIGraph *igraph, int i, int j)
{
  // Node 
  igraph->spSize[i] += igraph->spSize[j];

  // Parametric update
  for (int f = 0; f < igraph->feats->ncols; ++f) {
    iftMatrixRowPointer(igraph->feats, i)[f] +=
      iftMatrixRowPointer(igraph->feats, j)[f];
    iftMatrixRowPointer(igraph->seedFeats, i)[f] =
      iftMatrixRowPointer(igraph->feats, i)[f] / igraph->spSize[i];
  }

  // Geometric update
  for (int f = 0; f < igraph->pos->ncols; ++f) {
    iftMatrixRowPointer(igraph->pos, i)[f] +=
      iftMatrixRowPointer(igraph->pos, j)[f];
    iftMatrixRowPointer(igraph->seedPos, i)[f] =
      iftMatrixRowPointer(igraph->pos, i)[f] / igraph->spSize[i];
  }

  // Update adjancecy, can be one-sided due to union-find representation
  iftSet *jointAdj = iftSetUnion(igraph->nodeAdj[i], igraph->nodeAdj[j]);
  iftDestroySet(&(igraph->nodeAdj[i]));
  igraph->nodeAdj[i] = jointAdj;
}

extern void iftRemoveLabelGaps(int *labelArray, int size);

iftImage * prgConvertToLabelImage(iftSuperpixelIGraph *igraph, int *parent)
{
  iftImage *res = iftCopyImage(igraph->refImg);
  for (int p = 0; p < res->n; ++p) {
    int node = res->val[p] - 1;
    if (node < 0)
      continue;
    res->val[p] = prgFind(parent, node);
  }

  iftRemoveLabelGaps(res->val, res->n);
  return res;
}

float prgMergeEnergyDelta(iftSuperpixelIGraph *igraph, int i, int j, int *parent)
{
  float internalEdge = prgArcWeight(igraph, i, j);
  float removedEdges = 0.0f;
  float addedEdges = 0.0f;

  // -- Energy decrease from removed edges
  // Note that internal edge is counted twice
  prgNormalizeAdj(&(igraph->nodeAdj[i]), parent);
  for (iftSet *s = igraph->nodeAdj[i]; s != NULL; s = s->next)
    removedEdges += prgArcWeight(igraph, i, s->elem);
  prgNormalizeAdj(&(igraph->nodeAdj[j]), parent);
  for (iftSet *s = igraph->nodeAdj[j]; s != NULL; s = s->next)
    removedEdges += prgArcWeight(igraph, j, s->elem);
  // Internal edge was counted twice, separate it
  removedEdges -= (2.0f * internalEdge);

  // -- Energy increase from added edges
  // Find adjacency of potential merged node
  //iftSet *jointAdj = iftSetUnion(igraph->nodeAdj[i], igraph->nodeAdj[j]);
  iftSet *jointAdj = iftSetConcat(igraph->nodeAdj[i], igraph->nodeAdj[j]);
  iftRemoveSetElem(&jointAdj, i);
  iftRemoveSetElem(&jointAdj, j);
  int count = iftSetSize(jointAdj);

  // Use virtual node to hold features of potential merged node
  int k = igraph->nNodes; // not an actual node
  for (int f = 0; f < igraph->feats->ncols; ++f) {
    iftMatrixRowPointer(igraph->seedFeats, k)[f] =
      iftMatrixRowPointer(igraph->feats, i)[f] +
      iftMatrixRowPointer(igraph->feats, j)[f];
    iftMatrixRowPointer(igraph->seedFeats, k)[f] /= 
      (igraph->spSize[i] + igraph->spSize[j]);
  }
  for (int f = 0; f < igraph->pos->ncols; ++f) {
    iftMatrixRowPointer(igraph->seedPos, k)[f] =
      iftMatrixRowPointer(igraph->pos, i)[f] +
      iftMatrixRowPointer(igraph->pos, j)[f];
    iftMatrixRowPointer(igraph->seedPos, k)[f] /= 
      (igraph->spSize[i] + igraph->spSize[j]);
  }

  // Get the actual weights
  while (jointAdj != NULL)
    addedEdges += prgArcWeight(igraph, k, iftRemoveSet(&jointAdj));

  float neighAvgDelta = (addedEdges - removedEdges) / ((float)count);

  return neighAvgDelta;
}

void prgInitSeedMx(iftSuperpixelIGraph *igraph)
{
  iftDestroyMatrix(&(igraph->seedFeats));
  iftDestroyMatrix(&(igraph->seedPos));
  igraph->seedFeats = iftCreateMatrix(igraph->feats->ncols, igraph->feats->nrows + 1);
  igraph->seedPos = iftCreateMatrix(igraph->pos->ncols, igraph->pos->nrows + 1);

  // Extra node is intentionally ignored
  for (int i = 0; i < igraph->feats->n; ++i)
    igraph->seedFeats->val[i] = igraph->feats->val[i];
  for (int i = 0; i < igraph->pos->n; ++i)
    igraph->seedPos->val[i] = igraph->pos->val[i];
}

void prgNormalizeAdj(iftSet **s, int *parent)
{
  if (s == NULL || *s == NULL)
    return;

  // Convert everything to representatives
  for (iftSet *aux = *s; aux != NULL; aux = aux->next)
    aux->elem = prgFind(parent, aux->elem);
  // Remove duplicates
  for (iftSet *aux = *s; aux != NULL; aux = aux->next)
    iftRemoveSetElem(&(aux->next), aux->elem);
}

void iftUpdateDHeapElem(iftDHeap *H, int node, double val)
{
  if (H->pos[node] < 0)
    iftError("Node is not in the heap.", "iftUpdateDHeapElem");

  double aux = H->value[node];
  H->value[node] = val;

  if (val < aux) {
    if (H->removal_policy == MINVALUE)
      iftGoUpDHeap(H, H->pos[node]);
    else
      iftGoDownDHeap(H, H->pos[node]);
  } else {
    if (H->removal_policy == MINVALUE)
      iftGoDownDHeap(H, H->pos[node]);
    else
      iftGoUpDHeap(H, H->pos[node]);
  }
}

