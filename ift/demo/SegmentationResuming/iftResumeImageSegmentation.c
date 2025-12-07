//
// Created by tvspina on 1/18/16.
//

#include "ift.h"

int main(int argc, char **argv) {
    iftImage *orig = NULL, *basins = NULL;
    iftImage *input_label_image = NULL;
    iftAdjRel *A = NULL;
    iftDict *json = NULL;
    iftLabeledSet *optional_seeds = NULL;
    double alpha = 0.0;

    char *input_image_file = NULL;
    char *input_label_file = NULL;
    char *output_dir = NULL;
    char *json_config_file = NULL;
    char *gradient_type = NULL;
    char *img_filename = NULL;
    char *img_basename = NULL;
    char *optional_seed_file = NULL;

    /*--------------------------------------------------------*/

    int MemDinInicial, MemDinFinal;
    MemDinInicial = iftMemoryUsed();

    /*--------------------------------------------------------*/

    if(argc < 5)
        iftError("usage: %s <original image(.ppm,.pgm,.scn)> <input segmentation(.pgm,.scn)> <resuming_config_file.json> <output directory> [mask gradient combination alpha] [optional input seed set]", "main", argv[0]);

    input_image_file = iftCopyString(argv[1]);
    input_label_file = iftCopyString(argv[2]);
    json_config_file = iftCopyString(argv[3]);
    output_dir = iftCopyString(argv[4]);

    if(argc > 5)
        alpha = atof(argv[5]);

    if(argc > 6)
        optional_seed_file = iftCopyString(argv[6]);

    if(!iftFileExists(json_config_file))
        iftError("Please create a resuming_config.json file!", "main");

    json = iftReadJson(json_config_file);
    iftPrintDict(json);

    img_filename = iftFilename(input_image_file, NULL);
    img_basename = iftBasename(img_filename);

    iftMakeDir(output_dir);

    /** Reading images **/
    orig = iftReadImageByExt(input_image_file);
    input_label_image = iftReadImageByExt(input_label_file);

    if(optional_seed_file != NULL) {
        if(!iftFileExists(optional_seed_file)) {
            iftError("Optional seed set file %s does not exist!", "main");
        } else {
            optional_seeds = iftReadSeeds(optional_seed_file, orig);
        }
    }
    /* Computing gradient */
    if(iftIs3DImage(orig))
        A = iftSpheric(1.0);
    else
        A = iftCircular(1.5);

    gradient_type = iftGetStrValFromDict("gradient_type", json);

    if(iftCompareStrings(gradient_type, "IMAGE_BASINS"))
        basins = iftImageBasins(orig, A);
    else if(iftCompareStrings(gradient_type, "IMAGE_GRADIENT_MAGNITUDE"))
        basins = iftImageGradientMagnitude(orig, A);
    else if(iftCompareStrings(gradient_type, "RESP_SYSTEM_SPECIFIC"))
        basins = iftRespSystemGradient(orig, A);
    else
        iftError("Invalid gradient type. Please choose among IMAGE_BASINS, IMAGE_GRADIENT_MAGNITUDE, or RESP_SYSTEM_SPECIFIC", "main");

    if(!iftAlmostZero(alpha)) {
        iftImage *tmp = NULL, *mask_basins = NULL, *stretched = NULL;

        stretched   = iftLinearStretch(input_label_image, iftMinimumValue(input_label_image), iftMaximumValue(input_label_image),
                                       iftMinimumValue(basins), iftMaximumValue(basins));
        mask_basins = iftImageBasins(stretched, A);

        tmp = iftLinearCombination(mask_basins, basins, alpha);

        iftDestroyImage(&basins);
        iftDestroyImage(&mask_basins);
        iftDestroyImage(&stretched);

        basins = tmp;
    }

    iftResumeImageSegmentation(orig, input_label_image, basins, A, json, optional_seeds, output_dir, img_basename);

    iftFree(input_image_file);
    iftFree(input_label_file);
    iftFree(output_dir);
    iftFree(gradient_type);
    iftFree(img_filename);
    iftFree(img_basename);
    iftFree(json_config_file);
    if(optional_seed_file != NULL) iftFree(optional_seed_file);

    iftDestroyDict(&json);
    iftDestroyImage(&orig);
    iftDestroyImage(&basins);
    iftDestroyImage(&input_label_image);
    iftDestroyAdjRel(&A);
    iftDestroyLabeledSet(&optional_seeds);

    /* ---------------------------------------------------------- */

    MemDinFinal = iftMemoryUsed();
    iftVerifyMemory(MemDinInicial, MemDinFinal);

    return 0;
}
