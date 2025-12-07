#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **gt_path, char **mbb_img_path,
                        char **out_score_basename);
float iftDistanceBetweenMBBCenters(iftBoundingBox bb1, iftBoundingBox bb2, float voxel_size);
float iftPercOfLostObjectVoxels(  iftImage *gt_img,   iftImage *mbb_img, int label);
float iftBgVoxelsRatio(  iftImage *gt_img,   iftImage *mbb_img, int label);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory arguments
    char *gt_path = NULL;
    char *mbb_img_path        = NULL;
    char *out_score_basename       = NULL;

    iftGetRequiredArgs(args, &gt_path, &mbb_img_path, &out_score_basename);

    timer *t1 = iftTic();

    puts("- Reading Test Label Image");
    iftImage *gt_img = iftReadImageByExt(gt_path);

    puts("- Min Bounding Box Image - Brain Substructure Localization");
    iftImage *mbb_img = iftReadImageByExt(mbb_img_path);

    if (!iftIsVoxelSizeEqual(gt_img, mbb_img)) {
        iftError("Test Label Image and Min Bounding Box Image with different voxel sizes\n" \
                 "(%f, %f, %f) != (%f, %f, %f)", "main",
                 gt_img->dx, gt_img->dy, gt_img->dz,
                 mbb_img->dx, mbb_img->dy, mbb_img->dz);
    }
    printf("voxel size: %f\n", gt_img->dx);


    iftIntArray *labels     = iftGetObjectLabels(gt_img);
    iftBoundingBox *gt_mbbs = iftMinLabelsBoundingBox(gt_img, labels, NULL);
    iftBoundingBox *mbbs    = iftMinLabelsBoundingBox(mbb_img, labels, NULL);


    puts("- Computing Scores");
    puts("Score 1: Distance (in mm) between the mbb centers");
    puts("Score 2: Percentage of Lost Object's Voxels");
    puts("");

    for (int o = 0; o < labels->n; o++) {
        printf("* Object: %d\n", labels->val[o]);
        printf("gt_mbbs.begin: %d, %d, %d\n", gt_mbbs[o].begin.x, gt_mbbs[o].begin.y, gt_mbbs[o].begin.z);
        printf("gt_mbbs.end: %d, %d, %d\n", gt_mbbs[o].end.x, gt_mbbs[o].end.y, gt_mbbs[o].end.z);

        float score1 = iftDistanceBetweenMBBCenters(gt_mbbs[o], mbbs[o], gt_img->dx);
        printf("Score 1 = %f mm\n", score1);

        float score2 = iftPercOfLostObjectVoxels(gt_img, mbb_img, labels->val[o]);
        printf("Score 2 = %f\n", score2);

        float score3 = iftBgVoxelsRatio(gt_img, mbb_img, labels->val[o]);
        printf("Score 3 = %f\n\n", score3);
        

        // storing the scores into file
        char out_score_csv[512];
        sprintf(out_score_csv, "%s_%d.csv", out_score_basename, labels->val[o]);
        FILE *score_fp = fopen(out_score_csv, "a+");
        fprintf(score_fp, "%s;%f;%f;%f\n", gt_path, score1, score2, score3);
        fclose(score_fp);
    }
    iftDestroyIntArray(&labels);
    iftFree(mbbs);
    iftFree(gt_mbbs);


    if (iftDictContainKey("--output-dir", args, NULL)) {
        const char *out_dir = iftGetStrValFromDict("--output-dir", args);
        if (!iftDirExists(out_dir))
            iftMakeDir(out_dir);

        char *test_filename = iftFilename(gt_path, NULL);
        char *out_path = iftJoinPathnames(2, out_dir, test_filename);
    
        puts("- Writing Label Image (Test Label Image + Min Bounding Box Image)");
        iftImage *test_bin = iftThreshold(gt_img, 1, IFT_INFINITY_INT, 1);
        iftImage *mbb_bin  = iftThreshold(mbb_img, 1, IFT_INFINITY_INT, 1);
        iftImage *out_img = iftAdd(test_bin, mbb_bin);
        iftWriteImageByExt(out_img, out_path);
        iftDestroyImage(&test_bin);
        iftDestroyImage(&mbb_bin);
        iftDestroyImage(&out_img);

        iftFree(test_filename);
        iftFree(out_path);
    }



    iftDestroyImage(&gt_img);
    iftDestroyImage(&mbb_img);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));


    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(gt_path);
    iftFree(mbb_img_path);
    iftFree(out_score_basename);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Computes the Accuracy of the Bran Substructure Localization.\n" \
        "- There are 3 measures:\n" \
        "1) Distance (in mm) between the center of the computed min bounding box and the true " \
        "min bounding box;\n" \
        "2) Percentage of object's voxels out of the computed min bounding box;\n" \
        "3) Percentage of background voxels inside the computed min bounding box";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-t", .long_name = "--ground-truth-label-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Ground Truth (Test Label Image) with the objects of interest."},
        {.short_name = "-l", .long_name = "--localizer-bounding-box-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Localizer Image with the bounding boxes from the target objects."},
        {.short_name = "-s", .long_name = "--output-score-file-basename", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output basename file where the scores will be saved. It will be created " \
                               "a csv score file for each object. The scores will be appended into files in a new row, " \
                               "by following the order: test_img_path,score1,score2"},
        {.short_name = "-o", .long_name = "--output-dir", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Output Directory to save the test objects over the localizer bounding boxes.\n" \
                                "The output image will have the same filename of the input test image."},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **gt_path, char **mbb_img_path,
                        char **out_score_basename) {
    *gt_path            = iftGetStrValFromDict("--ground-truth-label-image", args);
    *mbb_img_path       = iftGetStrValFromDict("--localizer-bounding-box-image", args);
    *out_score_basename = iftGetStrValFromDict("--output-score-file-basename", args);

    char *parent_dir = iftParentDir(*out_score_basename);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- Ground Truth Image: \"%s\"\n", *gt_path);
    printf("- Min Bounding Box Image: \"%s\"\n", *mbb_img_path);
    printf("- Output Score CSV: \"%s\"\n", *out_score_basename);
    puts("-----------------------");
}


float iftDistanceBetweenMBBCenters(iftBoundingBox bb1, iftBoundingBox bb2, float voxel_size) {
    iftVoxel c1 = iftBoundingBoxCenterVoxel(bb1);
    iftVoxel c2 = iftBoundingBoxCenterVoxel(bb2);

    printf("test_obj_mbb_center: %d, %d, %d\n", c1.x, c1.y, c1.z);
    printf("localizer_obj_mbb_center: %d, %d, %d\n\n", c2.x, c2.y, c2.z);

    return (iftVoxelDistance(c1, c2) * voxel_size);
}


float iftPercOfLostObjectVoxels(  iftImage *gt_img,   iftImage *mbb_img, int label) {
    iftImage *test_obj_bin = iftThreshold(gt_img, label, label, 1); // extract the object with label <label> and binarize them
    iftImage *mbb_obj_bin  = iftThreshold(mbb_img, label, label, 1); // extract the object with label <label> and binarize them

    iftImage *inv_mbb_obj_bin = iftComplement(mbb_obj_bin); // gets the complement from the target object's bounding box
    iftImage *lost_obj_img = iftMask(test_obj_bin, inv_mbb_obj_bin); // image with the lost object's voxels
    
    int n_test_obj_voxels = iftCountObjectSpels(test_obj_bin, 1); // count the number of voxels from the target object in binary image
    int n_test_lost_voxels = iftCountObjectSpels(lost_obj_img, 1); // count the number of voxels from the target object in binary image

    float perc_lost_obj_voxels = (n_test_lost_voxels * 1.0) / n_test_obj_voxels;

    // printf("n_test_obj_voxels = %d\n", n_test_obj_voxels);
    // printf("n_test_lost_voxels = %d\n", n_test_lost_voxels);
    // printf("perc_lost_obj_voxels = %f\n", perc_lost_obj_voxels);

    // iftWriteImageByExt(test_obj_bin, "out/test_obj_bin_%d.hdr", label);
    // iftWriteImageByExt(mbb_obj_bin, "out/mbb_obj_bin_%d.hdr", label);
    // iftWriteImageByExt(inv_mbb_obj_bin, "out/inv_mbb_obj_bin_%d.hdr", label);
    // iftWriteImageByExt(lost_obj_img, "out/lost_obj_img_%d.hdr", label);

    iftDestroyImage(&test_obj_bin);
    iftDestroyImage(&mbb_obj_bin);
    iftDestroyImage(&inv_mbb_obj_bin);
    iftDestroyImage(&lost_obj_img);

    return perc_lost_obj_voxels;
}


float iftBgVoxelsRatio(  iftImage *gt_img,   iftImage *mbb_img, int label) {
    iftImage *mbb_obj_bin  = iftThreshold(mbb_img, label, label, 1); // extract the object with label <label> and binarize them

    iftImage *test_obj_bin = iftThreshold(gt_img, label, label, 1); // extract the object with label <label> and binarize them
    iftImage *inv_test_obj_bin = iftComplement(test_obj_bin); // gets the complement from the target object's bounding box
    iftImage *gt_mbb_obj_bin = iftCreateImageFromImage(test_obj_bin);

    iftBoundingBox gt_obj_mbb = iftMinBoundingBox(test_obj_bin, NULL); // object mbb from the Ground Truth
    
    for (int z = gt_obj_mbb.begin.z; z <= gt_obj_mbb.end.z; z++)
        for (int y = gt_obj_mbb.begin.y; y <= gt_obj_mbb.end.y; y++)
            for (int x = gt_obj_mbb.begin.x; x <= gt_obj_mbb.end.x; x++)
                iftImgVal(gt_mbb_obj_bin, x, y, z) = 1;


    // Grount Truth
    iftImage *gt_mbb_bg = iftMask(gt_mbb_obj_bin, inv_test_obj_bin);
    int n_gt_mbb_bg_voxels = iftCountObjectSpels(gt_mbb_bg, 1); // count the number of voxels from the target object in binary image
    // iftWriteImageByExt(gt_mbb_bg, "tmp/gt_mbb_bg_%d.hdr", label);
    // printf("n_gt_mbb_bg_voxels: %d\n", n_gt_mbb_bg_voxels);
    
    iftImage *mbb_bg = iftMask(mbb_obj_bin, inv_test_obj_bin);
    int n_true_mbb_bg_voxels = iftCountObjectSpels(mbb_bg, 1); // count the number of voxels from the target object in binary image
    // iftWriteImageByExt(mbb_bg, "tmp/mbb_bg_%d.hdr", label);
    // printf("n_true_mbb_bg_voxels: %d\n", n_true_mbb_bg_voxels);

    float bg_voxels_ratio = (n_true_mbb_bg_voxels * 1.0) / n_gt_mbb_bg_voxels;


    iftDestroyImage(&test_obj_bin);
    iftDestroyImage(&inv_test_obj_bin);
    iftDestroyImage(&mbb_obj_bin);
    iftDestroyImage(&gt_mbb_obj_bin);

    return bg_voxels_ratio;
}
/*************************************************************/










