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

void iftTestSeedObjectModel(  iftFileSet *test_img_files,   iftFileSet *gt_img_files, char *model_basename,
                            const char *output_dir, iftJson *segmentation_config_json);

/*************************************************************/

int main(int argc, char *argv[]){
    char *model_basename = NULL;
    iftFileSet *test_img_files = NULL; // loads the image pathnames
    iftFileSet *tmp_files = NULL;
    iftFileSet *gt_img_files = NULL;
    iftJson *segmentation_config_json = NULL;

    char *test_img_entry = NULL;
    char *test_img_dirname = NULL;
    char *gt_img_entry = NULL;
    char *output_dir = NULL;
    char *segmentation_config_file = NULL;

    /*--------------------------------------------------------*/

    void *trash = malloc(1);
    struct mallinfo info;
    int MemDinInicial, MemDinFinal;
    free(trash);
    info = mallinfo();
    MemDinInicial = info.uordblks;

    /*--------------------------------------------------------*/

    if (argc != 6)
        iftError("%s <(test image directory) OR (test_image_pathnames.csv)> <(gt directory) OR (gt_pathnames.csv)> <model basename> <segmentation_config_file.json> <output dir>",
                 "main", argv[0]);

    test_img_entry = iftCopyString(argv[1]);
    gt_img_entry = iftCopyString(argv[2]);
    model_basename = iftCopyString(argv[3]);
    segmentation_config_file = iftCopyString(argv[4]);
    output_dir = iftCopyString(argv[5]);

    test_img_dirname = iftParentDir(test_img_entry);
    // Moving to the experiment directory
    if(!iftCompareStrings(test_img_dirname, "") && !iftCompareStrings(test_img_dirname, "."))
        chdir(test_img_dirname);

    test_img_files = iftLoadFileSetFromDirOrCSV(test_img_entry, true);

    tmp_files = iftLoadFileSetFromDirOrCSV(gt_img_entry, true);

    gt_img_files = iftFilterFilesBySubset(test_img_files, tmp_files, "", "");
    iftDestroyFileSet(&tmp_files);

    if(!iftFileExists(segmentation_config_file))
        iftError("Please provide a segmentation configuration file with parameters for gradient estimation!", "main");

    segmentation_config_json = iftReadJson(segmentation_config_file);

    timer *t1 = iftTic();

    iftTestSeedObjectModel(test_img_files, gt_img_files, model_basename, output_dir, segmentation_config_json);

    timer *t2 = iftToc();
    iftPrintFormattedTime(iftCompTime(t1, t2));
    puts("Done...");

    // DESTROYERS
    iftDestroyFileSet(&test_img_files);
    iftDestroyFileSet(&gt_img_files);
    iftDestroyJson(&segmentation_config_json);

    free(model_basename);
    free(test_img_entry);
    free(test_img_dirname);
    free(gt_img_entry);
    free(output_dir);
    free(segmentation_config_file);

    /* ---------------------------------------------------------- */

    info = mallinfo();
    MemDinFinal = info.uordblks;
    if (MemDinInicial!=MemDinFinal)
        printf("\n\nDynamic memory was not completely deallocated (%d, %d)\n",
               MemDinInicial,MemDinFinal);

    return 0;
}


void iftTestSeedObjectModel(  iftFileSet *test_img_files,   iftFileSet *gt_img_files, char *model_basename,
                            const char *output_dir, iftJson *segmentation_config_json) {
    int i;
    double avg_dice = 0.0, avg_assd = 0.0;
    iftSeedObjectModel *model = NULL;

    iftMakeDir(output_dir);

    model = iftReadSeedObjectModel(model_basename);

    for (i = 0; i < test_img_files->nfiles; i++) {
        double dice, assd;

        int msps_nscales;
        int msps_min_displacement = 1;
        int msps_max_displacement;

        char *test_img_basename = iftFilename(test_img_files->files[i]->pathname, "");
        char *output_basename = NULL;
        char *output_filename = NULL;
        const char *ext = iftFileExt(test_img_files->files[i]->pathname);

        iftImage *test_img = NULL, *gt = NULL;
        iftObjectModelProb *prob = NULL;
        iftLabeledSet *objects = NULL, *borders = NULL;
        iftVoxel best_pos;
        iftJson *json = NULL;

        test_img = iftReadImageByExt(test_img_files->files[i]->pathname);
        gt = iftReadImageByExt(gt_img_files->files[i]->pathname);

        if(!iftFileExists("msps.json"))
            iftError("Please create an msps.json file with MSPS' configuration", "main");

        json = iftReadJson("msps.json");

        fprintf(stderr, "Evaluating: %s (GT %s)\n", test_img_files->files[i]->pathname, gt_img_files->files[i]->pathname);

        timer *t1 = iftTic();

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

        iftMSPSObjectModelSearchByTranslationSingleCenter(prob, model->model_center, msps_nscales,
                                                          msps_min_displacement,
                                                          msps_max_displacement, &best_pos);

        fprintf(stderr,"Best position (%d %d %d) (Started from %d %d %d)\n", best_pos.x, best_pos.y, best_pos.z,
                model->model_center.x, model->model_center.y, model->model_center.z);

        iftMSPSDelineateObjectWithModel(prob, best_pos, &objects, &borders);

        iftDestroyLabeledSet(&objects);
        iftDestroyLabeledSet(&borders);

        output_basename = iftConcatStrings(3, test_img_basename, ".", ext);
        output_filename = iftJoinPathnames(2, output_dir, output_basename);
        iftWriteImageByExt(prob->delineation_data->fst->label, output_filename);

        dice = iftDiceSimilarity(gt, prob->delineation_data->fst->label);
        assd = iftASSD(gt, prob->delineation_data->fst->label) * gt->dx * gt->dy * gt->dz;

        fprintf(stderr, "Dice: %lf\n", dice);
        fprintf(stderr, "ASSD: %lf\n", assd);

        avg_assd += assd;
        avg_dice += dice;

        timer *t2 = iftToc();
        iftPrintFormattedTime(iftCompTime(t1, t2));

        iftDestroyImage(&test_img);
        iftDestroyImage(&gt);
        iftDestroyImage(&prob->delineation_data->fst->img);
        iftDestroyAdjRel(&prob->delineation_data->fst->A);
        iftDestroyImageForest(&prob->delineation_data->fst);
        iftDestroyOMRecognitionDataWatershedMeanCut(&prob->recognition_data);
        iftDestroyOMDelineationAlgorithmDataWatershed(&prob->delineation_data);
        iftDestroyObjectModelProb(&prob);
        iftDestroyJson(&json);
        free(test_img_basename);
        free(output_basename);
        free(output_filename);
        free(segmentation_config_json);
    }

    avg_assd /= MAX(1.0, test_img_files->nfiles);
    avg_dice /= MAX(1.0, test_img_files->nfiles);

    fprintf(stderr, "Average Dice: %lf\n", avg_dice);
    fprintf(stderr, "Average ASSD: %lf\n", avg_assd);

    iftDestroySeedObjectModel(&model);
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

        const char *ext = iftFileExt(img_path);

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
