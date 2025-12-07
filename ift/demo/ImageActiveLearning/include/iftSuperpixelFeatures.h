/**
 * @file
 * @author Felipe Lemes Galvao
 * @date July 6, 2017
 *
 * @details General utility to extract features from superpixel
 *   segmentations. All methods work for both 2D and 3D unless
 *   explicitly noticed. To avoid using the non-standard terminology
 *   'supervoxel', we only refer to 'superpixels' and 'pixels' even
 *   in methods that work in 3D.
 * @details We assume superpixel segmentations are represented by
 *  a \c iftImage containing the superpixel label map.
 * @details For now results are always stored into an \c iftMatrix
 *   where the row r corresponds to the superpixel of label r.
 */

#ifndef _IFT_SUPERPIXEL_FEATURES_H_
#define _IFT_SUPERPIXEL_FEATURES_H_

#include <ift.h>
#include "iftBoVW.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Distance function for covariance matrices.
 * @author Felipe Lemes Galvao
 * @date August 2, 2017
 *
 * @details Covariance metric as used on 
 *   https://doi.org/10.1007/11744047_45
 * @details In general, both inputs must represent square matrices
 *   that are symmetric or Hermitian, and \c f2 must be positive
 *   definite. Covariance matrices by definition meet those conditions.
 * @details This distance function is computationally expensive so
 *   it is expected that the input covariance matrices are relatively
 *   small.
 *
 * @param[in] f1 Input feature vector 1.
 * @param[in] f2 Input feature vector 2.
 * @param[in] alpha Ignored (to match iftArcWeightFun signature).
 * @param[in] n Feature vector sizes.
 */
float iftCovarianceDistance(float *f1, float *f2, float *alpha, int n);

/**
 * @brief Use medoid of some color space as superpixel feature.
 * @author Felipe Lemes Galvao
 * @date August 1, 2017
 *
 * @todo Fill in params and preconditions.
 */
iftMatrix * iftComputeSuperpixelFeaturesByColorSpaceMedoid(  iftImage *superpixelLabelMap,   iftImage *img, iftColorSpace colorSpace);

/**
 * @brief Use medoid of pixel values as superpixel feature.
 * @author Felipe Lemes Galvao
 * @date June 27, 2017
 *
 * @details For each superpixel, calculate the mean (average) value of
 *   its pixels' feature vectors (i.e. \c mimg values) and then use
 *   the pixel feature vector closest to the mean as the superpixel
 *   feature vector.
 * @details The returned \c iftDataSet is owned by the caller.
 * @see iftDestroyDataSet()
 *
 * @param[in] mimg Base \c iftMImage.
 * @param[in] superpixelLabelMap Some superpixel segmentation of mimg.
 * @return Data matrix where each row corresponds to the corresponding
 * @pre \c mimg and \c superpixelLabelMap are not NULL and share the
 *      same spatial domain.
 * @pre \c superpixelLabelMap does not contain label gaps.
 */
iftMatrix * iftComputeSuperpixelFeaturesByMImageMedoid(  iftMImage *mimg,   iftImage *superpixelLabelMap);

/**
 * @brief Use the BoVW histogram as superpixel feature.
 * @author Felipe Lemes Galvao
 * @date June 28, 2017
 *
 * @details For each superpixel, build a normalized BoVW histogram
 *   where each input is the feature vector of one pixel.
 * @see iftBoVW
 * @details The returned \c iftMatrix is owned by the caller.
 * @see iftDestroyMatrix()
 *
 * @param[in] mimg Base MImage.
 * @param[in] superpixelLabelMap Some superpixel segmentation of mimg.
 * @param[in] bag Pointer to the BoVW descriptor.
 * @param[in] k Soft assignment k. k = 1 for hard assignment.
 * @return The (nSuperpixels x nBins) feature matrix of normalized
 *         histograms.
 * 
 * @pre \c mimg and \c superpixelLabelMap are not NULL and share the
 *      same spatial domain.
 * @pre \c superpixelLabelMap does not contain label gaps.
 * @pre \c bag is not NULL and is compatible with mimg feature vector
 */
iftMatrix * iftComputeSuperpixelFeaturesByColorBoVW(  iftMImage *mimg,   iftImage *superpixelLabelMap,   iftBoVW *bag, int k);

/**
 * @brief Use superpixel covariance matrix as feature.
 * @author Felipe Lemes Galvao
 * @date August 2, 2017
 *
 * @details The result rows represent the covariance matrix as
 *  a 1D array. The original shape is easy to recover as we are
 *  dealing with a square matrix.
 * @details We assume a concatenation of pixel features have been
 *  computed beforehand.
 * @details The returned \c iftMatrix is owned by the caller.
 * @see iftDestroyMatrix()
 * @see iftCovarianceDistance()
 *
 * @param[in] superpixelLabelMap Some superpixel segmentation of mimg.
 * @param[in] pixelFeats Pixel level features over the same domain.
 * @return The (nSuperpixels x (pixelFeats->ncols)^2) feature matrix
 *         of convariance matrix features.
 *
 * @pre All parameters are not NULL.
 * @pre There are features for all pixels
 *      (i.e. pixelFeats->nrows == superpixelLabelMap->n)
 */
iftMatrix * iftComputeSuperpixelFeaturesByRegionCovarianceMatrix(  iftImage *superpixelLabelMap,   iftMatrix *pixelFeats);

/**
 * @brief Computes color histogram for superpixels.
 * @author Felipe Lemes Galvao
 * @date January 5, 2017
 *
 * @details There is a very similar function that creates a iftDataSet
 *   of superpixel color histogram features. This one exists more as a
 *   supplement for the superpixel contextual features.
 *
 * @return Each row is a superpixel histogram (NOT normalized).
 */
iftMatrix * iftComputeSuperpixelFeaturesByColorHistogram(  iftImage *img,   iftImage *labelMap, int binsPerColor);

/**
 * @brief Computes BIC histogram for superpixels.
 * @author Felipe Lemes Galvao
 * @date January 5, 2017
 *
 * @details There is a very similar function that creates a iftDataSet
 *   of superpixel BIC histogram features. This one exists more as a
 *   supplement for the superpixel contextual features.
 */
iftMatrix * iftComputeSuperpixelFeaturesByBICHistogram(  iftImage *img,   iftImage *labelMap, int binsPerColor);

/**
 * @brief Computes contextual histogram for superpixels.
 * @author Felipe Lemes Galvao
 * @date January 5, 2017
 *
 * @details By contextual we mean, for each superpixel we compute
 *   a histogram with the pixels from superpixels at EXACTLY 
 *   \c neighDist of the base superpixel.
 * @details It works for any kind of base superpixel histogram
 *   (e.g. conventional color histogram, BIC).
 * @details All involved histograms MUST NOT be normalized.
 *
 * @param[in] spHistFeats Pre-computed histogram for each superpixel.
 */
iftMatrix * iftComputeSuperpixelFeaturesByNeighborHistogram(  iftSuperpixelIGraph *igraph, iftMatrix *spHistFeats, int neighDist);

/**
 * @brief Computes color histogram at multiple superpixel distances.
 * @author Felipe Lemes Galvao
 * @date January 5, 2017
 */
iftMatrix * iftComputeSuperpixelFeaturesByContextualColorHistogram(  iftSuperpixelIGraph *igraph,   iftImage *img, int binsPerColor, int order);

/**
 * @brief Computes BIC histogram at multiple superpixel distances.
 * @author Felipe Lemes Galvao
 * @date January 5, 2017
 */
iftMatrix * iftComputeSuperpixelFeaturesByContextualBICHistogram(  iftSuperpixelIGraph *igraph,   iftImage *img, int binsPerColor, int order);

#ifdef __cplusplus
}
#endif

#endif // _IFT_SUPERPIXEL_FEATURES_H_
