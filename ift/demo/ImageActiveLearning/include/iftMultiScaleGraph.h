/**
 * @file
 * @author Felipe Lemes Galvao
 * @date September, 2018
 *
 * @brief A graph for mixing multiple segmentation scales.
 *
 * @details A graph for using multiple segmentation scales
 *   (e.g. created by RISF) simultaneously. Graph starts as a
 *   regular superpixel adjacency graph (RAG) in the coarsest
 *   scale and allows the conversion of any node to its finer
 *   scale children while keeping the active graph consistent.
 * @details Current implementation is a proof of concept so
 *   various design decisions favor ease of implementation
 *   over efficiency. This means it might not be fast enough
 *   for large images (e.g. 3D).
 */

#ifndef _IFT_MULTISCALE_GRAPH_
#define _IFT_MULTISCALE_GRAPH_

#include <ift.h>
#include "iftSuperpixelFeatures.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ift_multiscale_sp_graph {
  int nScales;
  int *nNodesInScale;
  int **activeNodeFlagInScale;
  iftSet ***nodeAdjInScale; /* 4 rightmost bits for scale, rest for node */
  iftMatrix **featsInScale;
  iftImage **refImgInScale;
  int **labelInScale; /* Pre-allocated auxiliary data. */
  iftImage *markers;
} iftMultiScaleSpGraph;

/*
 * @brief Creates the multiscale graph from hierarchy of sp maps.
 * @author Felipe Lemes Galvao
 * @date September, 2018
 */
iftMultiScaleSpGraph *iftCreateMultiScaleGraph(int nScales, iftImage **spLabelImgInScale, iftImage *img);

/*
 * @brief Remove a superpixel and add its finer scale parts to graph.
 * @author Felipe Lemes Galvao
 * @date September, 2018
 */
void iftExpandMultiScaleGraphSp(iftMultiScaleSpGraph *graph, int scale, int sp);

/*
 * @brief Make graph consistent with existing markers/seeds.
 * @author Felipe Lemes Galvao
 * @date October, 2018
 *
 * @details Marker image is stored in the graph and assumed to have
 *   0 for unlabeled pixels and label = (value - 1) for
 *   remaining ones.
 */
void iftUpdateMultiScaleSpGraphByMarkers(iftMultiScaleSpGraph *graph);

/*
 * @brief Build label map out of next finer level.
 * @author Felipe Lemes Galvao
 * @date October, 2018
 */
iftImage *iftMultiScaleSpGraphFinerLabelMap(iftMultiScaleSpGraph *graph);

#ifdef __cplusplus
}
#endif

#endif /* _IFT_MULTISCALE_GRAPH_ */
