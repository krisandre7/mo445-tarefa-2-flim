#include <ift.h>

/**********************************************************************
															DEFINITIONS
**********************************************************************/
#define PROG_DESC \
				"This program samples the seeds considering the strategy of overseeding the "\
				"background, while sampling the centroid of the components which area percentage "\
				"(compared with the overall sum) is greater than a certain threshold. It considers an "\
				"(grayscale/binary) object saliency map as reference."
#define IMG_DESC \
				"Input image"
#define K_DESC \
				"Number of seed points (x > 0)"
#define OUT_RAD_DESC \
				"Background erosion radius (x >= 0.0)"
#define THR_OBJSM_DESC \
				"Threshold value for the object saliency map (0 <= x <= 1)"
#define THR_SEED_DESC \
				"Threshold value for seed selection (0 <= x <= 1)"
#define MASK_DESC \
				"Mask indicating the reachable pixels"
#define OBJSAL_DESC \
				"Object saliency map"
#define OUT_DESC \
				"Output image containing the seeds"
				
/**********************************************************************
																METHODS
**********************************************************************/
/**
* @brief Extracts the arguments given in the command line
* @author Felipe Belem
* @date August 2018
*
* @param argc - Number of arguments
* @param argv - Arguments provided by the command line
*
* @return An iftDict containing the parsed args
**/
iftDict *iftGetArgs
(int argc, const char *argv[]);

/**
* @brief Assigns and validates the extracted required arguments from the dictionary
* @author Felipe Belem
* @date August 2018
*
* @param args - Dictionary containing the arguments
* @param img_path - Path to the input image
* @param k - Number of seeds
* @param objsal_path - Path to the object saliency map image
* @param out_path - Path to the output image
**/
void iftGetAndValReqArgs
(  iftDict *args, char **img_path, int *k, char **objsal_path, char **out_path);

/**
* @brief Assigns and validates the extracted optional arguments from the dictionary
* @author Felipe Belem
* @date November 2018
*
* @param args - Dictionary containing the arguments
* @param mask_path - Path to the mask image
* @param out_rad - Background erosion radius
* @param thr_objsm - Threshold for the object saliency map
* @param thr_seed - Threshold of seed displacement criteria
**/
void iftGetAndValOptArgs
(  iftDict *args, char **mask_path, float *out_rad, float *thr_objsm, float *thr_seed);

/**
* @brief Performs a proportional GRID sampling on the object saliency map
*				 in which the background, will receive a higher amount of seeds,
*				 whereas each object will receive a single seed in its geometric center.
* @author Felipe Belem
* @date September 2018
*
* @param objsm - Object saliency map
* @param mask - Mask indicating the reachable pixels
* @param k - Number of seeds
* @param out_rad - Background erosion radius
* @param thr_objsm - Threshold value for the object saliency map
* @param thr_seed - Threshold value for the seed selection
**/
iftImage *iftCENTERSampling
(iftImage *objsm, iftImage *mask, int k, float out_rad, float thr_objsm, float thr_seed );

/**
* @brief Samples the centroid of each component in the labeled image if,
*				 and only if, its area is greater than the threshold.
* @author Felipe Belem
* @date November 2018
*
* @param label - Labeled image
* @param mask - Mask indicating the reachable pixels
* @param thresh - Threshold value for the object saliency map
**/
iftSet *iftMultiLabelCenterSamplingOnMaskByArea
(  iftImage *label, iftImage *mask, float thresh);