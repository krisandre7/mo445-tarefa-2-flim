//
// Created by spyder on 1/20/16.
//

#include "iftSeedObjectModel.h"

/************************** HEADERS **************************/
void iftValidateInputs(const char *moving_img_entry, const char *out_dir);

/**
 * Checks if all paths from the File Set are images and has the same file extension, returning such file extension.
 */
const char *iftValidateImageFileSet(iftFileSet *img_files);

void iftTestSeedObjectModel(  iftFileSet *test_img_files,   iftFileSet *gt_img_files, iftSeedObjectModel *seeds,
                            const char *output_dir, iftJson *segmentation_config_json);

/*************************************************************/


//
// Created by tvspina on 1/18/16.
//
#include "ift.h"
#include "iftSeedObjectModel.h"

/************************** HEADERS **************************/
void iftValidateInputs(const char *moving_img_entry, const char *out_dir);

/**
 * Checks if all paths from the File Set are images and has the same file extension, returning such file extension.
 */
const char *iftValidateImageFileSet(iftFileSet *img_files);

/*************************************************************/


int main(int argc, char *argv[]){
    iftSeedObjectModel *model = NULL;
    iftFileSet *test_img_files = NULL; // loads the image pathnames
    iftFileSet *tmp_files = NULL;
    iftFileSet *gt_img_files = NULL;
    iftJson *json = NULL;

    char *model_basename = NULL;
    char *test_img_entry = NULL;
    char *test_img_dirname = NULL;
    char *gt_img_entry = NULL;
    char *config_file = NULL;
    char *output_dir = NULL;

    /*--------------------------------------------------------*/

    void *trash = malloc(1);
    struct mallinfo info;
    int MemDinInicial, MemDinFinal;
    free(trash);
    info = mallinfo();
    MemDinInicial = info.uordblks;

    /*--------------------------------------------------------*/


    if (argc != 6)
        iftError("%s <(test image directory) OR (test_image_pathnames.csv)> <(gt directory) OR (gt_pathnames.csv)> <model basename> <json segmentation configuration file> <output dir>",
                 "main", argv[0]);

    test_img_entry = iftCopyString(argv[1]);
    gt_img_entry = iftCopyString(argv[2]);
    model_basename = iftCopyString(argv[3]);
    config_file = iftCopyString(argv[4]);
    output_dir = iftCopyString(argv[5]);


    test_img_dirname = iftParentDir(test_img_entry);
    // Moving to the experiment directory
    if(!iftCompareStrings(test_img_dirname, "") && !iftCompareStrings(test_img_dirname, "."))
        chdir(test_img_dirname);
    
    test_img_files = iftLoadFileSetFromDirOrCSV(test_img_entry, true);

    tmp_files = iftLoadFileSetFromDirOrCSV(gt_img_entry, true);

    json = iftReadJson(config_file);
    iftPrintJson(json);

    gt_img_files = iftFilterFilesBySubset(test_img_files, tmp_files, "", "");
    iftDestroyFileSet(&tmp_files);

    timer *t1 = iftTic();

    model = iftReadSeedObjectModel(model_basename);

    iftTestSeedObjectModel(test_img_files, gt_img_files, model, output_dir, json);

    timer *t2 = iftToc();
    iftPrintFormattedTime(iftCompTime(t1, t2));
    puts("Done...");

    // DESTROYERS
    iftDestroySeedObjectModel(&model);
    iftDestroyFileSet(&test_img_files);
    iftDestroyJson(&json);
    free(model_basename);
    free(test_img_entry);
    free(test_img_dirname);
    free(gt_img_entry);
    free(config_file);
    free(output_dir);

    /* ---------------------------------------------------------- */

    info = mallinfo();
    MemDinFinal = info.uordblks;
    if (MemDinInicial!=MemDinFinal)
        printf("\n\nDynamic memory was not completely deallocated (%d, %d)\n",
               MemDinInicial,MemDinFinal);

    return 0;
}

/************************** SOURCES **************************/
void iftValidateInputs(const char *moving_img_entry, const char *out_dir) {
    if ((!iftFileExists(moving_img_entry)) && (!iftDirExists(moving_img_entry)))
        iftError("Invalid Pathname for the Moving Image Set Entry: \"%s\"... Try a valid Directory or a CSV file",
                 "iftValidateInputs", moving_img_entry);
    if (iftFileExists(out_dir))
        iftError("Output Directory \"%s\" is actually a File", "iftValidateInputs", out_dir);
    else if (!iftDirExists(out_dir))
        iftMakeDir(out_dir);
}


const char *iftValidateImageFileSet(iftFileSet *img_files) {
    if (img_files == NULL)
        iftError("Image File Set is NULL", "iftValidateImageFileSet");

    const char *ext = NULL;
    if (img_files->nfiles != 0) {
        char *img_path = img_files->files[0]->pathname;

        if (!iftIsImageFile(img_path))
            iftError("File %s is not an Image (.pgm, .pp, .scn)", "iftValidateImageFileSet", img_path);

        ext = iftFileExt(img_path);

        for (int i = 1; i < img_files->nfiles; i++) {
            img_path = img_files->files[i]->pathname;
            if (!iftIsImageFile(img_path))
                iftError("File %s is not an Image (.pgm, .pp, .scn)", "iftValidateImageFileSet", img_path);

            const char *ext2 = iftFileExt(img_path);
            if (!iftCompareStrings(ext, ext2))
                iftError("Image Set with different extensions: %s, %s", "iftValidateImageFileSet", ext, ext2);
        }
    }
    else
        iftError("There is no Images", "iftValidateImageFileSet");

    return ext;
}
/*************************************************************/

void iftTestSeedObjectModel(  iftFileSet *test_img_files,   iftFileSet *gt_img_files, iftSeedObjectModel *model,
                            const char *output_dir, iftJson *segmentation_config_json) {
    int i;
    double avg_dice = 0.0;

    double gradient_combination_alpha;
    char *gradient_type = NULL;

    gradient_type = iftGetJString(segmentation_config_json, "gradient_type");
    gradient_combination_alpha = iftGetJDouble(segmentation_config_json, "gradient_combination_alpha");

    iftMakeDir(output_dir);

    for(i = 0; i < test_img_files->nfiles; i++) {
        double dice;
        const char *ext = iftFileExt(test_img_files->files[i]->pathname);
        char *ext_with_dot = iftConcatStrings(2, ".", ext);
        char *test_img_basename = iftFilename(test_img_files->files[i]->pathname, ext_with_dot);
        char *output_basename = NULL;
        char *output_filename = NULL;

        iftImage *test_img = NULL;
        iftImage *gt = NULL;
        iftImage *basins = NULL;
        iftImage *label = NULL;
        iftAdjRel *A = NULL;

        test_img = iftReadImageByExt(test_img_files->files[i]->pathname);
        gt = iftReadImageByExt(gt_img_files->files[i]->pathname);

        fprintf(stderr, "Evaluating: %s (GT %s)\n", test_img_files->files[i]->pathname, gt_img_files->files[i]->pathname);

        timer *t1 = iftTic();

        if (iftIs3DImage(test_img)) {
            A = iftSpheric(1.0);
        } else {
            A = iftCircular(1.5);
        }

        if(iftCompareStrings(gradient_type, "IMAGE_BASINS"))
            basins = iftImageBasins(test_img, A);
        else if(iftCompareStrings(gradient_type, "IMAGE_GRADIENT_MAGNITUDE"))
            basins = iftImageGradientMagnitude(test_img, A);
        else if(iftCompareStrings(gradient_type, "RESP_SYSTEM_SPECIFIC"))
            basins = iftRespSystemGradient(test_img, A);
        else if(iftCompareStrings(gradient_type, "GAUSSIAN_ENHANCED"))
            basins = iftGaussianEnhancedImageGradient(test_img, A, model->intensities_mode, model->intensities_mode_deviation, gradient_combination_alpha);
        else
            iftError("Invalid gradient type. Please choose among IMAGE_BASINS, IMAGE_GRADIENT_MAGNITUDE, RESP_SYSTEM_SPECIFIC, or GAUSSIAN_ENHANCED", "main");


        label = iftWatershed(basins, A, model->seeds, NULL);

        timer *t2 = iftToc();
        iftPrintFormattedTime(iftCompTime(t1, t2));

        dice = iftDiceSimilarity(gt, label);

        fprintf(stderr, "Dice: %lf\n", dice);

        avg_dice += dice;

        output_basename = iftConcatStrings(2, test_img_basename, ext_with_dot);
        output_filename = iftJoinPathnames(2, output_dir, output_basename);
        iftWriteImageByExt(label, output_filename);
        free(output_basename);
        free(output_filename);

        output_basename = iftConcatStrings(3, test_img_basename, "_basins", ext_with_dot);
        output_filename = iftJoinPathnames(2, output_dir, output_basename);
        iftWriteImage(basins, output_filename);

        iftDestroyImage(&test_img);
        iftDestroyImage(&label);
        iftDestroyImage(&basins);
        iftDestroyImage(&gt);
        iftDestroyAdjRel(&A);
        free(test_img_basename);
        free(output_basename);
        free(output_filename);
        free(ext_with_dot);
    }

    avg_dice /= MAX(1.0, test_img_files->nfiles);

    fprintf(stderr, "Average Dice: %lf\n", avg_dice);


    free(gradient_type);
}