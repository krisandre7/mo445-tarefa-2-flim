//
// Created by tvspina on 1/19/16.
//

#ifndef IFT_IFTSEEDOBJECTMODEL_H
#define IFT_IFTSEEDOBJECTMODEL_H

#include "iftCommon.h"
#include "iftImage.h"
#include "iftFImage.h"
#include "iftFile.h"
#include "iftRepresentation.h"
#include "iftDataSet.h"
#include "iftMatrix.h"
#include "iftJson.h"
#include "iftGraphics.h"
#include "iftSimilarity.h"
#include "iftObjectModels.h"

/** By TVS object search using a seed-based model **/


/** By TVS **/

typedef struct iftSeedObjectModel {
    iftFImage *model;
    iftImage *candidate_seed_label;
    iftLabeledSet *seeds;
    iftVoxel model_center;
    iftBoundingBox uncert_region_bbox;
    iftVector start, end;

    double selected_threshold;

    double intensities_mode; // Statistical mode of the voxel intensities among all training images
    double intensities_mode_deviation; // Standard deviation from the mode

} iftSeedObjectModel;

iftSeedObjectModel *iftCreateSeedObjectModel(iftFImage *model, iftImage *candidate_seed_label, iftVoxel model_center,
                                             iftBoundingBox uncert_region_bbox, iftVector search_start,
                                             iftVector search_end);
void iftDestroySeedObjectModel(iftSeedObjectModel **model);

iftSeedObjectModel *iftCreateSeedObjectModelByAverage(const char *ref_seed_basename, iftFileSet *seed_img_files,
                                                      iftFileSet *label_img_files);

void iftDetermineSeedsForSeedObjectModel(iftSeedObjectModel *model, double thresh);

void iftDetermineTextureInformationForSeedObjectModel(iftSeedObjectModel *model,   iftFileSet *orig_img_files,
                                                        iftFileSet *label_img_files, int normalization_value);

/**
 * @brief Writes a seed object model to an output_basename.
 *
 * @author Thiago Vallin Spina
 *
 * @param model Float image containing the membership scores.
 * @param output_basename The basename to be used for the files. *
 * @param ext_with_dot The model's output extension.
 */
void iftWriteSeedObjectModel(iftSeedObjectModel *model, const char *output_basename);

/**
 * @brief Reads a seed object model from an input_basename.
 *
 * @author Thiago Vallin Spina
 *
 * @param model Float image containing the membership scores.
 * @param input_basename The basename to be used to read the files.
 * @param ext_with_dot The model's input extension.
 */
iftSeedObjectModel* iftReadSeedObjectModel(char *input_basename);


/**
 * @brief Converts a seed object model into an object model problem for search with MSPS. The seeds are converted
 * to adjacency relations centered at the model's center (i.e., the reference label geometric center) for translation.
 *
 * @author Thiago Vallin Spina
 *
 * @param model The input seed object model
 * @param obj_label The delineation label.
 * @param test_image The image to be segmented.
 *
 * @return The object model problem.
 */
iftObjectModelProb *iftSeedObjectModelToObjectModelProb(iftSeedObjectModel *model, int obj_label, iftImage *test_image,
                                                        iftJson *segmentation_config_json);

char *iftLongestCommonSuffix(const char *str0, const char *str1);
char *iftLongestCommonPrefix(const char *str0, const char *str1);

iftImage *iftRespSystemGradient(iftImage *img, iftAdjRel *A);

void iftDrawSeeds(iftImage *img, iftImage *seed_image, iftColorTable *cmap);

#endif //IFT_IFTSEEDOBJECTMODEL_H
