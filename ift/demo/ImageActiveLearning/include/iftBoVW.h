/**
 * @file
 * @brief Bag of Visual Words (BoVW) interface
 * @author Felipe Lemes Galvao
 * @date June 26, 2017
 *
 * @details Given the superpixel segmentation of one or more images,
 *   the actual algorithm for building a BoVW has the following steps:
 * @arg Select a prototype for each relevant superpixel;
 * @arg Apply some clustering algorithm over all superpixel prototypes;
 * @arg Select and store cluster prototypes (e.g. OPF roots).
 * 
 * @details Each cluster prototype defines a bin of a BoVW histogram.
 *   Features for a new superpixel are then calculated by building the
 *   BoVW histogram with all its pixels, where each pixel falls into
 *   the bin corresponding to its closest cluster prototypes.
 *   The normalized BoVW histogram becomes the output feature vector.
 *
 * @details Basic usage:
 * @arg Create customized \c iftBoVWOptions options (optional)
 * @arg Use any of the functions prefixed with \c iftCreateBoVW
 *      to generate an \c iftBoVW
 * @arg Use any of the functions prefixed with \c iftComputeBoVWFeats
 *      to compute superpixel features 
 *
 * @details Some possibilities for future work:
 * @arg Updating an existing \c iftBoVW with new data;
 * @arg Soft assignment during histogram calculation (e.g. softmax over knn);
 * @arg Alternate distance function between pixel features;
 * @arg More variations of currently supported options.
 */

// TODO Separate superpixel functionality so that BoVW works on arbitrary datasets

#ifndef _IFT_BOVW_H_
#define _IFT_BOVW_H_

#include <ift.h>

#ifdef __cplusplus
extern "C" {
#endif

// Temporary for debug
extern iftSet *resRoots;

/**
 * @brief BoVW descriptor
 * @author Felipe Lemes Galvao
 * @date June 26, 2017
 *
 * @details The BoVW descriptor stores \c nFeatsOutput representative
 *   feature vectors of size \c nFeatsInput. Those are used to
 *   represent a histogram with \c nFeatsOuput bins, where new pixels
 *   fall into the bin of the closest representative feature vector.
 */
typedef struct ift_bovw {
  /** Original feature space used to create BoVW. */
  int nFeats;
  /** Number of histogram bins. */
  int nBins;
  /** nBins x nFeats data matrix with each bin prototype. */
  iftMatrix *feats;
} iftBoVW;

/**
 * @brief Creates BoVW descriptor using OPF clustering.
 * @author Felipe Lemes Galvao
 * @date June 27, 2017
 *
 * @details The returned \c iftBoVW is owned by the caller.
 * @see iftDestroyBoVW()
 *
 * @param[in] mx Data matrix where each row is a a feature vector.
 * @param[in] kmaxPercent Percentage of samples used as OPF kmax.
 * @param[in] graphCutFun OPF graph cut function (e.g. iftNormalizedCut)
 * @return The new BoVW descriptor.
 */
iftBoVW * iftCreateBoVWByOPFClustering(iftMatrix *mx, float kmaxPercent, iftKnnGraphCutFun graphCutFun);

/**
 * @brief Creates BoVW descriptor using Kmeans clustering.
 * @author Felipe Lemes Galvao
 * @date August 2, 2017
 *
 * @details The returned \c iftBoVW is owned by the caller.
 * @see iftDestroyBoVW()
 *
 * @param[in] mx Data matrix where each row is a a feature vector.
 * @param[in] k Number of k-means clusters.
 * @param[in] maxIters Max number of k-means iterations.
 * @param[in] minImprovement Threshold for k-means convergence.
 * @return The new BoVW descriptor.
 */
iftBoVW * iftCreateBoVWByKmeans(iftMatrix *mx, int k, int maxIters, float minImprovement);

/**
 * @brief Destroy BoVW
 * @author Felipe Lemes Galvao
 * @date June 27, 2017
 *
 * @param[in,out] bag Pointer to BoVW to be destroyed.
 */
void iftDestroyBoVW(iftBoVW **bag);

/**
 * @brief Hard assignment of a feature vector into a BoVW histogram.
 * @author Felipe Lemes Galvao
 * @date June 28, 2017
 *
 * @details A hard assigment is the standard histogram behavior of
 *   adding 1 to a single bin, which is the closest bin representative
 *   in the BoVW context.
 *
 * @param[in] bag The BoVW descriptor.
 * @param[in] input The feature vector defining the assignment.
 * @param[in,out] histogram The BoVW histogram to be updated.
 *
 * @pre All parameters are not NULL and have matching dimensions.
 */
void iftBoVWHardAssignment(  iftBoVW *bag,   float *input, float *histogram);

/**
 * @brief Soft assignment of a feature vector into a BoVW histogram.
 * @author Felipe Lemes Galvao
 * @date June 28, 2017
 *
 * @todo Details of how soft assignment works
 * @details If k == 1, this function redirects to hard assignment.
 *
 * @param[in] bag The BoVW descriptor.
 * @param[in] input The feature vector defining the assignment.
 * @param[in,out] histogram The BoVW histogram to be updated.
 * @param[in] k Number of closest words considered for assignment.
 *
 * @pre All parameters are not NULL and have matching dimensions.
 * @pre \c k is positive.
 */
void iftBoVWSoftAssignment(  iftBoVW *bag,   float *input, float *histogram, int k);

#ifdef __cplusplus
}
#endif

#endif //_IFT_BOVW_H_

