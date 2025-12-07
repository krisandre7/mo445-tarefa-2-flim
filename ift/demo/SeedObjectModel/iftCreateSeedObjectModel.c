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

void iftWriteSeedObjectModel(iftSeedObjectModel *model, const char *output_basename);

/*************************************************************/

int main(int argc, char *argv[]){
    iftSeedObjectModel *model = NULL;

    // loads the image pathnames
    iftFileSet *tmp_files = NULL;
    iftFileSet *seed_img_files = NULL;
    iftFileSet *orig_img_files = NULL;
    iftFileSet *label_img_files = NULL;

    iftJson *ref_image_json = NULL;
    iftJson *seed_object_model_config_json = NULL;
    iftJson *segmentation_config_json = NULL;

    double thresh;
    int normalization_value;

    char *seed_img_entry = NULL;
    char *orig_img_entry = NULL;
    char *label_img_entry = NULL;
    char *output_basename = NULL;
    char *output_basedir = NULL;
    char *ref_image_json_file = NULL;
    char *ref_img_dirname = NULL;
    char *ref_image_file = NULL;
    char *ref_basename = NULL;
    char *seed_object_model_config_file = NULL;
    char *segmentation_config_file = NULL;
    const char *ext = NULL;
    const char *ext_img = NULL;
    char *tmp = NULL;

    /*--------------------------------------------------------*/

    void *trash = malloc(1);
    struct mallinfo info;
    int MemDinInicial, MemDinFinal;
    free(trash);
    info = mallinfo();
    MemDinInicial = info.uordblks;

    /*--------------------------------------------------------*/


    if (argc != 8)
        iftError("%s <reference_image_file.json> <(seed_image_directory) OR (seed_image_pathnames.csv)> <(original_image_pathnames.csv> OR (original_image directory)> <(label_image_pathnames.csv) OR (label_image_directory)> <seed_object_model_config_json.json> <segmentation_config_file.json> <output_basename>",
                 "main", argv[0]);

    ref_image_json_file = iftCopyString(argv[1]);
    seed_img_entry = iftCopyString(argv[2]);
    orig_img_entry = iftCopyString(argv[3]);
    label_img_entry = iftCopyString(argv[4]);
    seed_object_model_config_file = iftCopyString(argv[5]);
    segmentation_config_file = iftCopyString(argv[6]);
    output_basename = iftCopyString(argv[7]);

    /* Reading the reference image's filename from a Json file */
    if(!iftFileExists(ref_image_json_file))
        iftError("Json file containing information about the reference image unavailable!", "main");


    ref_img_dirname = iftParentDir(ref_image_json_file);
    // Moving to the experiment directory
    if(!iftCompareStrings(ref_img_dirname, "") && !iftCompareStrings(ref_img_dirname, "."))
        chdir(ref_img_dirname);

    ref_image_json = iftReadJson(ref_image_json_file);
    ref_image_file = iftGetJString(ref_image_json, "reference_image");

    ref_basename = iftFilename(ref_image_file, "");
    ext = iftFileExt(ref_basename);
    tmp = iftFilename(ref_basename, ext);
    // Ensuring that the reference seed file basename ends with .txt
    free(ref_basename);
    ref_basename = iftConcatStrings(2, tmp, "txt");

    if(!iftFileExists(segmentation_config_file))
        iftError("Json file containing information about the segmentation step unavailable (%s)!", "main",
                 segmentation_config_file);

    segmentation_config_json = iftReadJson(segmentation_config_file);
    normalization_value = iftGetJInt(segmentation_config_json, "normalization_value");

    if(!iftFileExists(seed_object_model_config_file))
        iftError("Json file containing information for creating the seed object model unavailable (%s)!", "main",
                 seed_object_model_config_file);

    seed_object_model_config_json = iftReadJson(seed_object_model_config_file);
    thresh = iftGetJDouble(seed_object_model_config_json, "model_threshold");

    // Reading the seed filenames
    seed_img_files = iftLoadFileSetFromDirOrCSV(seed_img_entry, true);

    // Reading the label dataset and filtering unwanted images by comparing the full set of read images with
    // the subset in seed_img_files
    tmp_files = iftLoadFileSetFromDirOrCSV(label_img_entry, true);

    ext_img = iftFileExt(tmp_files->files[0]->path);

    label_img_files = iftFilterFilesBySubset(seed_img_files, tmp_files, "txt", ext_img);
    iftDestroyFileSet(&tmp_files);

    // Reading the original image dataset and filtering unwanted images by comparing the full set of read images with
    // the subset in seed_img_files
    
    tmp_files = iftLoadFileSetFromDirOrCSV(orig_img_entry, true);

    ext_img = iftFileExt(tmp_files->files[0]->path);

    orig_img_files = iftFilterFilesBySubset(seed_img_files, tmp_files, "txt", ext_img);
    iftDestroyFileSet(&tmp_files);

    timer *t1 = iftTic();

    /* Creating a binary seed object model */
    model = iftCreateSeedObjectModelByAverage(ref_basename, seed_img_files, label_img_files);

    /* Determining seeds by thresholding the model's membership map */
    iftDetermineSeedsForSeedObjectModel(model, thresh*iftFMaximumValue(model->model));
    /* Determining intensity-based information for gradient estimation */
    iftDetermineTextureInformationForSeedObjectModel(model, orig_img_files, label_img_files, normalization_value);

    fprintf(stderr,"Mode %lf dev %lf\n", model->intensities_mode, model->intensities_mode_deviation);

    output_basedir = iftParentDir(output_basename);
    iftMakeDir(output_basedir);

    iftWriteSeedObjectModel(model, output_basename);

    timer *t2 = iftToc();
    iftPrintFormattedTime(iftCompTime(t1, t2));
    puts("Done...");

    // DESTROYERS
    iftDestroySeedObjectModel(&model);
    iftDestroyFileSet(&seed_img_files);
    iftDestroyFileSet(&orig_img_files);
    iftDestroyFileSet(&label_img_files);
    iftDestroyJson(&ref_image_json);
    iftDestroyJson(&segmentation_config_json);
    iftDestroyJson(&seed_object_model_config_json);

    free(ref_image_json_file);
    free(ref_basename);
    free(ref_img_dirname);
    free(ref_image_file);
    free(seed_img_entry);
    free(label_img_entry);
    free(orig_img_entry);
    free(output_basename);
    free(output_basedir);
    free(segmentation_config_file);
    free(seed_object_model_config_file);
    free(tmp);

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
    if (img_files->n != 0) {
        char *img_path = img_files->files[0]->path;

        if (!iftIsImageFile(img_path))
            iftError("File %s is not an Image (.pgm, .pp, .scn)", "iftValidateImageFileSet", img_path);

        ext = iftFileExt(img_path);

        for (int i = 1; i < img_files->n; i++) {
            img_path = img_files->files[i]->path;
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


