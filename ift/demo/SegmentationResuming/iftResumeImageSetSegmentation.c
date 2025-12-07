//
// Created by tvspina on 1/18/16.
//
#include "iftSegmentationResuming.h"

/************************** HEADERS **************************/
void iftValidateInputs(const char *moving_img_entry, const char *out_dir);

/**
 * Checks if all paths from the File Set are images and has the same file extension, returning such file extension.
 */
const char *iftValidateImageFileSet(iftFileSet *img_files);

void iftResumeSegmentationImageSet(  iftFileSet *img_files,   iftFileSet *gt_files,
                                     iftFileSet *optional_seed_files, iftJson *json, const char *out_dir);

/**
 * @brief Takes as input a set of image files to be resumed and a single seed filename and creates a file set
 * with the same seed file name for each img file set.
 *
 * @author Thiago Vallin Spina
 *
 * @param img_files The image file set to be resumed.
 * @param seed_filename The single seed filename to be reproduced.
 *
 * @return The file set with one file pathname for each file in <img_files> with the same path as seed_filename.
 */
iftFileSet *iftReproduceSingleSeedFileIntoFileSet(  iftFileSet *img_files, const char *seed_filename);

/*************************************************************/


int main(int argc, char *argv[]) {
    iftJson *json = NULL;
    iftFileSet *img_files = NULL;
    iftFileSet *gt_files = NULL, *tmp_files = NULL;
    iftFileSet *optional_seed_files = NULL;

    char *img_entry = NULL;
    char *img_dirname = NULL;
    char *gt_entry = NULL;
    char *optional_seeds_entry = NULL;
    char *out_dir = NULL;
    char *tmp = NULL;
    char *json_config_file = NULL;
    char *json_basename = NULL;
    const char *ext = NULL;
    char *ext_with_dot = NULL;

    /*--------------------------------------------------------*/

    void *trash = malloc(1);
    struct mallinfo info;
    int MemDinInicial, MemDinFinal;
    free(trash);
    info = mallinfo();
    MemDinInicial = info.uordblks;

    /*--------------------------------------------------------*/


    if (argc < 5)
        iftError("%s <(image_directory) OR (image_pathnames.csv)> <(ground truth directory) OR (ground_truth_pathnames.csv> <resuming_config_file.json> <output directory> [(optional_seed_files.csv) OR (optional seeds files directory) OR (optional_seed_file.txt]",
                 "main", argv[0]);


    img_entry = iftCopyString(argv[1]);
    gt_entry = iftCopyString(argv[2]);
    json_config_file = iftCopyString(argv[3]);
    out_dir = iftCopyString(argv[4]);

    if(argc > 5)
        optional_seeds_entry = iftCopyString(argv[5]);

    img_dirname = iftParentDir(img_entry);

    // Moving to the experiment directory
    if(!iftCompareStrings(img_dirname, "") && !iftCompareStrings(img_dirname, "."))
        chdir(img_dirname);

    json_basename = iftFilename(json_config_file, "");

    iftValidateInputs(img_entry, out_dir);

    // loads the image pathnames
    img_files = iftLoadFileSetFromDirOrCSV(img_entry, true);
    // Checks if all moving images are really images and have the same extension, returning this one.
    ext = iftValidateImageFileSet(img_files);
    ext_with_dot = iftConcatStrings(2, ".", ext); // it is gonna be used to obtain image basenames without extension

    // loads the ground truth pathnames
    tmp_files = iftLoadFileSetFromDirOrCSV(gt_entry, true);

    gt_files = iftFilterFilesBySubset(img_files, tmp_files, "", "");
    iftDestroyFileSet(&tmp_files);


    // loads the ground truth pathnames or reproduces a single path name as a file set
    if(optional_seeds_entry != NULL) {
        if (iftDirExists(optional_seeds_entry)) {
            tmp_files = iftLoadFileSetFromDirBySuffix(optional_seeds_entry, ".txt");
            optional_seed_files = iftFilterFilesBySubset(img_files, tmp_files, ext_with_dot, ".txt");
            iftDestroyFileSet(&tmp_files);
        } else {
            const char *ext_seed = iftFileExt(optional_seeds_entry);

            if(iftCompareStrings(ext_seed, ".csv")) {
                tmp_files = iftReadFileSetFromCSV(optional_seeds_entry, true);
                optional_seed_files = iftFilterFilesBySubset(img_files, tmp_files, ext_with_dot, ".txt");
                iftDestroyFileSet(&tmp_files);
            } else {
                optional_seed_files = iftReproduceSingleSeedFileIntoFileSet(img_files, optional_seeds_entry);
            }
        }
    }

    if (!iftFileExists(json_config_file))
        iftError("Please create a %s file!", "main", json_config_file);

    json = iftReadJson(json_config_file);

    // Displaying/copying configuration file
    iftPrintJson(json);
    tmp = iftJoinPathnames(2, out_dir,json_basename);
    iftWriteJson(json, tmp);
    free(tmp);

    timer *t1 = iftTic();

    iftResumeSegmentationImageSet(img_files, gt_files, optional_seed_files, json, out_dir);

    timer *t2 = iftToc();

    iftPrintFormattedTime(iftCompTime(t1, t2));
    puts("Done...");

    // DESTROYERS
    free(img_entry);
    free(img_dirname);
    free(gt_entry);
    free(out_dir);
    if(optional_seeds_entry != NULL) free(optional_seeds_entry);
    free(ext_with_dot);
    free(json_config_file);
    free(json_basename);
    iftDestroyFileSet(&img_files);
    iftDestroyFileSet(&gt_files);
    iftDestroyFileSet(&optional_seed_files);
    iftDestroyJson(&json);

    /* ---------------------------------------------------------- */

    info = mallinfo();
    MemDinFinal = info.uordblks;
    if (MemDinInicial!=MemDinFinal)
        printf("\n\nDynamic memory was not completely deallocated (%d, %d)\n",
               MemDinInicial,MemDinFinal);

    return 0;
}

iftFileSet *iftReproduceSingleSeedFileIntoFileSet(  iftFileSet *img_files, const char *seed_filename) {
    iftFileSet *copied_seed_set = NULL;

    copied_seed_set = iftCreateFileSet(img_files->nfiles);

    for(int i = 0; i < img_files->nfiles; i++) {
        copied_seed_set->files[i] = iftCreateFile(seed_filename);
    }

    return copied_seed_set;
}

void iftResumeSegmentationImageSet(  iftFileSet *img_files,   iftFileSet *gt_files,
                                     iftFileSet *optional_seed_files, iftJson *json, const char *out_dir) {

    double mode = 0.0;
    double dev = 0.0;
    double alpha = 0.0;
    int normalization_value;

    char *gradient_type = NULL;

    gradient_type = iftGetJString(json, "gradient_type");
    alpha         = iftGetJDouble(json, "gradient_combination_alpha");
    normalization_value = iftGetJInt(json, "normalization_value");

    if(iftCompareStrings(gradient_type, "GAUSSIAN_ENHANCED"))
        iftImageSetVoxelIntensityStatistics(img_files, gt_files, &mode, &dev, normalization_value);

    fprintf(stderr,"Mode %lf dev %lf\n", mode, dev);

    for (size_t i = 0; i < img_files->nfiles; i++) {
        char *img_path = img_files->files[i]->pathname;
        char *label_path = gt_files->files[i]->pathname;
        char *optional_seed_path = NULL;

        if(optional_seed_files != NULL)
            optional_seed_path = optional_seed_files->files[i]->pathname;

        fprintf(stderr,"Resuming segmentation for image %s (input label %s) (%02lu/%02lu)\n", img_path, label_path, i+1, img_files->nfiles);

        if (iftIsImageFile(img_path)) {
            const char *ext = NULL;
            char *ext_with_dot = NULL;
            char *img_basename = NULL;

            iftImage *orig = NULL, *basins = NULL;
            iftImage *label = NULL;
            iftAdjRel *A = NULL;
            iftLabeledSet *optional_seeds = NULL;

            ext = iftFileExt(img_path);
            ext_with_dot = iftConcatStrings(2, ".", ext);
            img_basename = iftFilename(img_path, ext_with_dot);

            /** Reading images **/
            orig = iftReadImageByExt(img_path);
            label = iftReadImage(label_path);

            if(optional_seed_path != NULL) {
                if(!iftFileExists(optional_seed_path)) {
                    iftError("Optional seed set file %s does not exist!", "main");
                } else {
                    optional_seeds = iftReadSeeds(optional_seed_path, orig);
                }
            }

            timer *t1 = iftTic();

            /* Computing gradient */
            if(iftIs3DImage(orig))
                A = iftSpheric(1.0);
            else
                A = iftCircular(1.5);

            if(iftCompareStrings(gradient_type, "IMAGE_BASINS"))
                basins = iftImageBasins(orig, A);
            else if(iftCompareStrings(gradient_type, "IMAGE_GRADIENT_MAGNITUDE"))
                basins = iftImageGradientMagnitude(orig, A);
            else if(iftCompareStrings(gradient_type, "RESP_SYSTEM_SPECIFIC"))
                basins = iftRespSystemGradient(orig, A);
            else if(iftCompareStrings(gradient_type, "GAUSSIAN_ENHANCED"))
                basins = iftGaussianEnhancedImageGradient(orig, A, mode, dev, alpha);
            else
                iftError("Invalid gradient type. Please choose among IMAGE_BASINS, IMAGE_GRADIENT_MAGNITUDE, RESP_SYSTEM_SPECIFIC, or GAUSSIAN_ENHANCED", "main");

            iftResumeImageSegmentation(orig, label, basins, A, json, optional_seeds, out_dir, img_basename);

            timer *t2 = iftToc();

            iftPrintFormattedTime(iftCompTime(t1, t2));

            iftDestroyImage(&orig);
            iftDestroyImage(&basins);
            iftDestroyImage(&label);
            iftDestroyAdjRel(&A);
            iftDestroyLabeledSet(&optional_seeds);

            free(img_basename);
            free(ext_with_dot);
        } else {
            iftWarning("Skipping: \"%s\" IS NOT an Image", "main", img_path);
        }
    }
    free(gradient_type);
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
