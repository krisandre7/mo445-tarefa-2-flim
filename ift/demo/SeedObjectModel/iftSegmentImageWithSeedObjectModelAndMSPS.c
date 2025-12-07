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

iftFileSet* iftFilterFilesExcludingSuffixes(iftFileSet *subset, iftFileSet *full_set, const char *suffix0, const char *suffix1);

/*************************************************************/

int main(int argc, char *argv[]){
    iftImage *test_img = NULL;
    iftSeedObjectModel *model = NULL;
    iftObjectModelProb *prob = NULL;
    iftLabeledSet *objects = NULL, *borders = NULL;
    iftVoxel best_pos;
    iftJson *json = NULL;
    iftJson *segmentation_config_json = NULL;

    int msps_nscales;
    int msps_min_displacement = 1;
    int msps_max_displacement;

    char *model_basename = NULL;
    char *test_image_file = NULL;
    char *segmentation_config_file = NULL;

    /*--------------------------------------------------------*/

    void *trash = malloc(1);
    struct mallinfo info;
    int MemDinInicial, MemDinFinal;
    free(trash);
    info = mallinfo();
    MemDinInicial = info.uordblks;

    /*--------------------------------------------------------*/

    if (argc != 4)
        iftError("%s <input model basename> <testimage path> <segmentation_config_file.json>", "main", argv[0]);

    model_basename = iftCopyString(argv[1]);
    test_image_file = iftCopyString(argv[2]);

    model = iftReadSeedObjectModel(model_basename);
    test_img = iftReadImageByExt(test_image_file);

    timer *t1 = iftTic();

    if(!iftFileExists("msps.json"))
        iftError("Please create an msps.json file with MSPS' configuration", "main");

    json = iftReadJson("msps.json");
    segmentation_config_json = iftReadJson(segmentation_config_file);

    prob = iftSeedObjectModelToObjectModelProb(model, 1, test_img, segmentation_config_json);

    msps_nscales = iftGetJInt(json, "msps_nscales");

    msps_max_displacement = MAX(1, model->end.x - model->start.x+1);
    msps_max_displacement = MAX(msps_max_displacement, model->end.y - model->start.y+1);
    msps_max_displacement = MAX(msps_max_displacement, model->end.z - model->start.z+1);

    msps_max_displacement = MAX(1, msps_max_displacement / MAX(1, msps_nscales+1));

    if(msps_nscales <= 1 || msps_max_displacement <= 1) {
        msps_max_displacement = msps_min_displacement = 1;
        msps_nscales = 1;
    }

    fprintf(stderr,"Max displacement %d\n", msps_max_displacement);
    fprintf(stderr,"Min displacement %d\n", msps_min_displacement);
    fprintf(stderr,"Nscales %d\n", msps_nscales);


    iftMSPSObjectModelSearchByTranslationSingleCenter(prob, model->model_center, msps_nscales, msps_min_displacement,
                                                      msps_max_displacement, &best_pos);

    fprintf(stderr,"best_pos (%d %d %d) (%d %d %d)\n", best_pos.x, best_pos.y, best_pos.z, model->model_center.x,
    model->model_center.y, model->model_center.z);

    iftMSPSDelineateObjectWithModel(prob, best_pos, &objects, &borders);

    iftDestroyLabeledSet(&objects);
    iftDestroyLabeledSet(&borders);

    iftWriteImage(prob->delineation_data->fst->label, "seg.scn");

    timer *t2 = iftToc();
    iftPrintFormattedTime(iftCompTime(t1, t2));
    puts("Done...");

    // DESTROYERS

    iftDestroyImage(&test_img);
    iftDestroyImage(&prob->delineation_data->fst->img);
    iftDestroyAdjRel(&prob->delineation_data->fst->A);
    iftDestroyImageForest(&prob->delineation_data->fst);
    iftDestroyOMRecognitionDataWatershedMeanCut(&prob->recognition_data);
    iftDestroyOMDelineationAlgorithmDataWatershed(&prob->delineation_data);
    iftDestroyObjectModelProb(&prob);
    iftDestroyJson(&json);
    iftDestroyJson(&segmentation_config_json);
    iftDestroySeedObjectModel(&model);

    free(model_basename);
    free(test_image_file);
    free(segmentation_config_file);


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
