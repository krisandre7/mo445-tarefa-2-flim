#include <ift.h>

/**********************************************************************
															DEFINITIONS
**********************************************************************/
#define PROG_DESC \
				"Performs the Object-based Iterative Spanning Forest superpixel segmentation. "\
				"It iteratively computes an Image Foresting Transform over improved seeds "\
				"with respect of a given object saliency map, and seed image (seed != 0). If "\
				"no object map is given, a blank image will be used."
#define IMG_DESC \
				"Input image"
#define SEED_DESC \
				"Seed image (seed != 0)"
#define OBJSAL_DESC \
				"Object saliency map"
#define MASK_DESC \
				"Mask indicating the reachable pixels"
#define ALPHA_DESC \
				"Compactness factor (x >= 0)"
#define BETA_DESC \
				"Boundary adherence factor (x >= 0)"
#define GAMMA_DESC \
				"Object saliency map relevance (x > 0)"
#define ITER_DESC \
				"Number of iterations for segmentation (x > 0)"
#define SMOOTH_DESC \
				"Number of iterations for smoothing (x >= 0)"
#define OUT_DESC \
				"Output image"
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
* @param seed_path - Path to the seed image
* @param out_path - Path to the output image
**/
void iftGetAndValReqArgs
(  iftDict *args, char **img_path, char **seed_path,  char **out_path);

/**
* @brief Assigns and validates the extracted optional arguments from the dictionary
* @author Felipe Belem
* @date August 2018
*
* @param args - Dictionary containing the arguments
* @param mask_path - Path to the mask of reachable pixels
* @param objsal_path - Path to the object saliency map
* @param alpha - Compactness factor
* @param beta - Boundary adherence factor
* @param gamma - Object saliency map importance factor
* @param iter - Number of iterations for segmentation
* @param smooth - Number of iterations for smoothing
**/
void iftGetAndValOptArgs
(  iftDict *args, char **mask_path, char **objsal_path, double *alpha, double *beta, double *gamma, int *iter, int *smooth);

/**
* @brief Creates an IGraph for the OISF computation
* @author Felipe Belem
* @date August 2018
*
* @param img - Original image
* @param mask - Mask of reachable pixels
* @param objsm - Object saliency map
*/
iftIGraph *iftInitOISFIGraph
(iftImage *img, iftImage *mask, iftImage *objsm);

/**
* @brief Extends a MImage object by adding an grayscale object saliency
* 			 map band.
* @author Felipe Belem
* @date August 2018
*
* @param mimg - MImage to be extended
* @param objsm - Object saliency map
**/
iftMImage *iftExtendMImageByObjSalMap
(iftMImage *mimg, iftImage* objsm );

/**
* @brief Computes the superpixels centers by its coordinates
* @author Felipe Belem
* @date August 2018
*
* @param igraph - Image graph
* @param seeds - Image seeds (!= 0)
* @param alpha - Compactness factor
* @param beta - Boundary adherence factor
* @param gamma - Object map confidence
* @param iters - Number of iterations for segmentation
**/
void iftOISFSegm
(iftIGraph *igraph, iftImage *seeds, double alpha, double beta, double gamma, int iters);

/**
* @brief Evaluates if there is a necessity for updating the seeds in accordance
*				 with the threshold established in the ISF paper. If it is possitive, the
*				 seeds are updated.
* @author Felipe Belem
* @date August 2018
*
* @param igraph - Image graph
* @param center - New seed indexes
* @param seed - Seed indexes
* @param nseeds - Number of seeds
*	@param trees_rm - Trees marked to be removed
* @param new_seeds - New seeds set
**/
void iftIGraphEvalAndAssignNewSeeds
(iftIGraph *igraph, int* center, int *seed, int nseeds, iftSet **trees_rm, iftSet **new_seeds );