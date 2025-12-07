/**
 * @file
 * @brief General RISF program.
 * @author Felipe Lemes Galvao
 *
 * @details Currently the superpixel features, distance function,
 *   number of iterations and number of smoothing iterations are
 *   hard-coded, but the code is ready to make them input parameters.
 */

#include <ift.h>
#include "iftSuperpixelFeatures.h"
#include "iftExperimentUtility.h"

int main(int argc, char* argv[])
{
  if (argc < 12) {
    printf("Usage: iftRISF <sampling> <seed-recomp> <path-cost> <image> <result> <nSuperpixels> <alpha> <beta> <nIters> <feats> <print-options> [prev-seg]\n");
    printf("sampling..... 1: Grid\n");
    printf("              2: Mixed\n");
    printf("              3: Random\n");
    printf("              4: Geodesic\n");
    printf("seed-recomp.. 1: Medoid (parametric)\n");
    printf("              2: Centroid (geometric)\n");
    printf("path-cost.... 1: Additive + Root\n");
    printf("              2: Additive + Mean\n");
    printf("image........ Input image.\n");
    printf("result....... Output segmentation path.\n");
    printf("nSuperpixels. Target number of superpixels.\n");
    printf("alpha........ Parametric distance linear weigth.\n");
    printf("beta......... Parametric distance exponential weigth.\n");
    printf("nIters....... Number of RISF iterations.\n");
    printf("feats........ 1: Color mean.\n");
    printf("              2: Color histogram\n");
    printf("print-options 0: Do not print anything\n");
    printf("              1: Human-readable stats (not implemented)\n");
    printf("              2: CSV stats\n");
    printf("prev-seg..... (OPTIONAL) Previous segmentation label map.\n");
    printf("              Runs normal ISF if not included.\n");
    return -1;
  }

  int printOpt = atol(argv[11]);
  bool isHierarchic = (argc >= 13);

  // Load Image
  iftImage *img = iftReadImageByExt(argv[4]);
  bool is3D = iftIs3DImage(img);

  // Prepare igraph reference image
  iftImage *refImg = NULL;
  if (isHierarchic) {
    // Load pre-computed superpixel segmentation (previous RISF level)
    refImg = iftReadImageByExt(argv[12]);
  } else {
    // Make superpixel graph act as a pixel-based one
    refImg = iftSelectImageDomain(img->xsize, img->ysize, img->zsize);
    for (int i = 0; i < refImg->n; ++i)
      refImg->val[i] = i + 1;
  }

  // Build igraph
  iftSuperpixelIGraph *igraph = iftInitSuperpixelIGraph(refImg);
  iftAdjRel *A = is3D ? iftSpheric(1.0) : iftCircular(1.0);
  if (isHierarchic)
    iftSetSuperpixelIGraphExplicitRAG(igraph, A);
  else
    iftSetSuperpixelIGraphImplicitAdjacency(igraph, A);

  // Set superpixel parametric features
  float alpha = atof(argv[7]);
  float beta = atof(argv[8]);
  int featOpt = atol(argv[10]);
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
    case 2: // 20^3-bins LAB histogram XiSquare dist
      pFeats = iftComputeSuperpixelFeaturesByColorHistogram(img, igraph->refImg,  20);
      for (int node = 0; node < pFeats->nrows; ++node)
 	iftNormalizeFeatures(iftMatrixRowPointer(pFeats, node), pFeats->ncols);
      iftSetSuperpixelIGraphFeatures(igraph, pFeats, iftDistance10, alpha, beta);
      break;
  }

  // Set RISF parameters
  iftISFSampling sampling = atol(argv[1]);
  iftISFSeedRecomp seedRecomp = atol(argv[2]);
  iftISFPathCost pathCost = atol(argv[3]);
  int nSuperpixels = atol(argv[6]);
  int nIters = atol(argv[9]);
  int nSmoothIters = 0; // For now, no smoothing over superpixels
  float smallSpThreshold = 0.00;
  if (sampling == IFT_ISF_GEODESIC_SAMPLING) {
    //nSuperpixels *= 1.2;
    //smallSpThreshold = 0.05; // Default = 0.05
  }
  iftISFOptions *optISF = iftInitISFOptions(nSuperpixels, nIters,
      nSmoothIters, smallSpThreshold, sampling, pathCost, seedRecomp);

  // Actually compute RISF
  timer *t1 = iftTic();
  iftImage *superpixelLabelMap = iftSuperpixelSegmentationByRISF(igraph, optISF, img);
  timer *t2 = iftToc();
  if (printOpt == 2)
    printf("%f", iftCompTime(t1, t2));

  // Write segmentation result to file
  iftWriteImageByExt(superpixelLabelMap, argv[5]);

  iftDestroyImage(&img);
  iftDestroyImage(&refImg);
  iftDestroySuperpixelIGraph(&igraph);
  iftDestroyAdjRel(&A);
  iftDestroyMatrix(&pFeats);
  iftDestroyISFOptions(&optISF);
  iftDestroyImage(&superpixelLabelMap);

  return 0;
}
