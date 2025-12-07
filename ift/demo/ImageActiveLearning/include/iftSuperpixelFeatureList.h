/**
 * @file
 * @brief Complement to RISF graph to hold multiple independent feats.
 * @author Felipe Lemes Galvao
 * @date July 27, 2017
 *
 * @details Currently not used as it adds too much complexity
 *   with little benefit.
 */

#ifndef _IFT_SUPERPIXEL_FEATURES_LIST_H_
#define _IFT_SUPERPIXEL_FEATURES_LIST_H_

#include <ift.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Choice of nodes during path cost distance computation.
 * @author Felipe Lemes Galvao
 * @date July 27, 2017
 */
typedef enum ift_pathcost_nodes {
  /** Distance between nodes from edge added to path. */
  IFT_EDGE_PATHCOST_NODES = 1,
  /** Distance between path root and new node. */
  IFT_ROOT_PATHCOST_NODES = 2,
  /** Distance between previous iter mean and new node. */
  IFT_MEAN_PATHCOST_NODES = 3
} iftPathCostNodes;

/**
 * @brief Single linked list with superpixel feature information.
 * @author Felipe Lemes Galvao
 * @date July 27, 2017
 */
typedef struct ift_superpixel_feature_list {
  /** Superpixel feature matrix. */
  iftMatrix *feats; /*< Size nSuperpixels x nFeats. */
  /** Distance function used to compare superpixel features. */
  iftArcWeightFun distFun; /*< float(float, float, float, int) */
  /** Alpha array used in iftArcWeightFun. */
  float *distAlpha; /*< Size nFeats. Default = { 1.0 } */
  /** Linear weight applied to obtained distances. */
  float alpha; /*< Default = 1.0 */
  /** Exponential weight applied to obtained distances. */
  float beta; /*< Default = 1.0 */
  /** Identifier for the origin of superpixel features. */
  int featId;
  /** Optional string identifying the list node. */
  char id[80];
  /** Identification of how path cost is actually calculated. */
  iftPathCostNodes pathCostNodes;
  /** Auxiliary features for path costs not using edge cost directly.  */
  iftMatrix *auxFeats;
  /** Single linked list next element (NULL if last one). */
  struct ift_superpixel_feature_list *next;
} iftSuperpixelFeatureList;

/**
 * @brief Insert superpixel feature node into list's beginning.
 * @author Felipe Lemes Galvao
 * @date July 27, 2017
 *
 * @param[in,out] listRef Target list.
 * @param[in] superpixelFeatures Some pre-computed superpixel features.
 * @param[in] distFun Distance function comparing features.
 *                    Can be NULL for merely informative features.
 * @param[in] identifier Optional name field for node.
 * @param[in] featId Helper for algorithms using a subset of features.
 * @param[in] pathCostNodes Choice of nodes for path cost.
 *
 * @pre \c listRef is not NULL (although its content can be NULL).
 * @pre \c superpixelFeatures is not NULL.
 */
void iftAddToSuperpixelFeatureList(iftSuperpixelFeatureList **listRef,   iftMatrix *superpixelFeatures, iftArcWeightFun distFun, const char *identifier,   int featId,   iftPathCostNodes pathCostNodes);

/**
 * @brief Destroy superpixel feature list.
 * @author Felipe Lemes Galvao
 * @date July 27, 2017
 */
void iftDestroySuperpixelFeatureList(iftSuperpixelFeatureList **listRef);

/**
 * @brief Sets the weight for distances computed with a given feature.
 * @author Felipe Lemes Galvao
 * @date July 28, 2017
 *
 * @param[in,out] listNode Target superpixel feature node.
 * @param[in] alpha Linear weight (alpha * d(s,t)).
 * @param[in] beta Exponential weight (d(s,t)^beta).
 */
void iftChangeSuperpixelFeatureWeight(iftSuperpixelFeatureList *listNode, float alpha, float beta);

/**
 * @brief Updates graph dependent auxiliary features (e.g. mean). 
 * @author Felipe Lemes Galvao
 * @date July 28, 2017
 * 
 * @param[in,out] listNode Target superpixel feature node.
 * @param[in] ann 
 */
void iftUpdateGraphBasedSuperpixelAuxFeats(iftSuperpixelFeatureList *listNode, iftForestAnnotationInfo *ann);

#ifdef __cplusplus
}
#endif

#endif //_IFT_SUPERPIXEL_FEATURES_LIST_H_
