#ifndef _IFT_EXPERIMENT_UTILITY_H_
#define _IFT_EXPERIMENT_UTILITY_H_

#include "ift.h"

#ifdef __cplusplus
extern "C" {
#endif

// Move to ift, documentation is with implementation
void iftDrawBordersSingleLabel(iftImage *img, iftImage *labelMap,   int label, iftColor YCbCr);

/**
 * @brief Print a couple segmentation metrics.
 * @author Felipe Lemes Galvao
 * @date July, 2017
 *
 * @param[in] segLabelMap The test segmentation result.
 * @param[in] gtLabelMap The segmentation groundtruth.
 *
 * @pre Both are non-NULL and share same domain.
 */
void iftPrintSuperpixelSegmentationMetrics(iftImage *segLabelMap, iftImage *gtLabelMap);

/**
 * @brief Overlays an image with its segmentation borders.
 * @author Felipe Lemes Galvao
 * @date July, 2017
 *
 * @param[in] img The base image.
 * @param[in] segLabelMap Some segmentation of \c img.
 * @param[in] YCbCr Resulting border color.
 *
 * @pre Both images are non-NULL and share the same domain.
 */
iftImage *iftOverlaySegmentationBorders(iftImage *img, iftImage *segLabelMap, iftColor YCbCr);

/**
 * @brief Counts number of pixels with each label in each superpixel.
 * @author Felipe Lemes Galvao
 * @date June 22, 2018
 *
 * @details Result matrix has nSuperpixel rows and nLabels columns.
 *
 * @param[in] spLabelMap Reference superpixel segmentation.
 * @param[in] baseLabelMap Label data being aggregated.
 * @param[in] If true, convert to [0,1] interval.
 *
 * @pre Both maps are non-NULL and share the same domain.
 */
iftMatrix *iftCountLabelsInSuperpixels(  iftImage *spLabelMap,   iftImage *baseLabelMap, bool isNormalized);

/**
 * @brief Assigns a segmentation label to each superpixel based on
 *   the groundtruth.
 * @author Felipe Lemes Galvao
 * @date August 12, 2017
 *
 * @param[in] segLabelMap Some superpixel segmentation.
 * @param[in] gtLabelMap Groundtruth segmentation of objects.
 *
 * @pre Both maps are non-NULL and share the same domain.
 */
iftImage *iftSuperpixelToMajoritySegmentation(  iftImage *segLabelMap,   iftImage *gtLabelMap);

/**
 * @brief Computes ASA (achievable segmentation accuracy).
 * @author Felipe Lemes Galvao
 * @date February 21, 2018
 *
 * @param[in] segLabelMap Some superpixel segmentation.
 * @param[in] gtLabelMap Groundtruth segmentation of objects.
 *
 * @pre Both are non-NULL and share the same domain.
 */
float iftAchievableSegmentationAccuracy(  iftImage *segLabelMap,   iftImage *gtLabelMap);

/**
 * @brief Computes Boundary Precision.
 * @author Felipe Lemes Galvao
 * @date February 21, 2018
 *
 * @param[in] segBorders Some superpixel segmentation borders.
 * @param[in] gtBorders Groundtruth segmentation borders.
 * @param[in] toleranceDist Max distance a \c segBorders pixel can be from 
 *                          a \c gtBorders pixel and be considered correct.
 *
 * @pre Both borders are non-NULL and share the same domain.
 * @pre \c toleranceDist is non-negative.
 * @pre \c segBorders is non-empty.
 */
float iftBoundaryPrecision(  iftImage *segBorders,   iftImage *gtBorders, float toleranceDist);

/**
 * @brief Computes Boundary Fscore.
 * @author Felipe Lemes Galvao
 * @date February 21, 2018
 *
 * @param[in] segBorders Some superpixel segmentation borders.
 * @param[in] gtBorders Groundtruth segmentation borders.
 * @param[in] toleranceDist Max distance a \c segBorders pixel can be from 
 *                          a \c gtBorders pixel and be considered correct.
 *
 * @pre Both borders are non-NULL and share the same domain.
 * @pre \c toleranceDist is non-negative.
 */
float iftBoundaryFScore(  iftImage *segBorders,   iftImage *gtBorders, float toleranceDist);

/**
 * @brief Encoding for dLog distance used in BIC.
 * @author Felipe Lemes Galvao
 * @date December 21, 2017
 *
 * @param val Value to be encoded.
 * @param isNormalized Flag to scale normalized values into [0,255].
 */
float iftDLogEncoding(float val, bool isNormalized);

int iftRandomSelectionWeightedByOrder(int n);
iftMatrix * iftSuperpixelClassificationResults(iftDataSet *Z,   iftImage *superpixelLabelMap,   iftImage *prevSegLabelMap);

/* // Original ISF functions as seen in demo/ImageGraph */
/* iftImage *iftExtract_ISF_GRID_ROOT_Superpixels(iftImage *img, int nsuperpixels, float alpha, float beta, int niters, int smooth_niters, int *nseeds, int *finalniters); */
/* iftImage *iftExtract_ISF_MIX_MEAN_Superpixels(iftImage *img, int nsuperpixels, float alpha, float beta, int niters, int smooth_niters, int *nseeds, int *finalniters); */
/* iftImage *iftExtract_ISF_GRID_MEAN_Superpixels(iftImage *img, int nsuperpixels, float alpha, float beta, int niters, int smooth_niters, int *nseeds, int *finalniters); */
/* iftImage *iftExtract_ISF_SP_MEAN_Superpixels(iftImage *img, iftImage *spLabelMap, float alpha, float beta, int niters, int smooth_niters, int *nseeds, int *finalniters); */

// Variations of the former using pre-computed seed sampling

/*
 * @brief Selects a single representative pixel for each label.
 * @author Felipe Lemes Galvao
 * @date October, 2018
 */
iftSet *iftFindRepresentativePixels(  iftImage *gtLabelMap);

/*
 * @brief Encapsulate distance from node to seed in the feature space..
 * @author Felipe Lemes Galvao
 * @date June, 2019
 */
float iftSpIGraphNodeToSeedFeatDist(iftSuperpixelIGraph *igraph, int node, int seed);

/*
 * @brief Encapsulate distance from node to seed in the geometric space.
 * @author Felipe Lemes Galvao
 * @date June, 2019
 */
float iftSpIGraphNodeToSeedSpaceDist(iftSuperpixelIGraph *igraph, int node, int seed);

/*
 * @brief SLIC-like post-processing.
 * @author Felipe Lemes Galvao
 * @date June, 2019
 *
 *   Turns each connected component into an unique label and relabel components below
 *     \c minSize (in pixels) to a neighbor's label.
 *   A similar function is implemented in iftSlic.c/h.
 */
iftImage *iftForceLabelMapConnectivity(iftImage *labelMap, int minSize);

/*
 * @brief Partitions igraph underlying image with Metis library.
 * @author Felipe Lemes Galvao
 * @date June, 2019
 * 
 *   Only implemented for explicit adjacencies.
 */
iftImage * iftBuildMetisInitPartition(iftSuperpixelIGraph *igraph, int nSp);

/*
 * @brief Reads CSV including header, if it exists.
 * @author Felipe Lemes Galvao
 * @date September, 2019
 *
 *   Direct edit of original iftReadCSV code.
 */
iftCSV *iftReadCSVWithHeader(const char *csv_pathname, const char separator, bool *has_header);

/*
 * @brief Extract superpixel sizes from label map.
 * @author Felipe Lemes Galvao
 * @date February, 2020
 * 
 * @details If nSp <= 0 it is calculated on the fly as
 *   iftMaximumValue(labelMap).
 */
int *iftGetSuperpixelSizes(  iftImage *labelMap, int nSp);

/*
 * @brief Get fraction of object overlap for each superpixel.
 * @author Felipe Lemes Galvao
 * @date February, 2020 
 *
 * @details If nSp <= 0 it is calculated on the fly as
 *   iftMaximumValue(labelMap). Only works if background
 *   is assigned the minimum label value in the ground truth.
 */
float *iftGetSuperpixelObjectOverlap(  iftImage *labelMap,   iftImage *gt, int nSp);

#ifdef __cplusplus
}
#endif

#endif // _IFT_EXPERIMENT_UTILITY_H_ 
