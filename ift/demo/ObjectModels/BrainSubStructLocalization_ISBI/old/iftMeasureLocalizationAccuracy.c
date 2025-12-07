#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **gt_path, char **mbb_json_path,
                        char **out_score_basename);
float iftDistanceBetweenMBBCenters(iftBoundingBox bb1, iftBoundingBox bb2, float voxel_size);
float iftPercOfLostObjectVoxels(  iftImage *gt_img, iftBoundingBox bb, int label);
float iftBgVoxelsError(  iftImage *gt_img, iftBoundingBox bb, int label);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory arguments
    char *gt_path            = NULL;
    char *mbb_json_path      = NULL;
    char *out_score_basename = NULL;

    iftGetRequiredArgs(args, &gt_path, &mbb_json_path, &out_score_basename);

    timer *t1 = iftTic();

    puts("- Reading Test Label Image");
    iftImage *gt_img = iftReadImageByExt(gt_path);

    puts("- Reading Min Bounding Box Image - Brain Substructure Localization");
    iftIntArray *bb_labels = NULL;
    iftBoundingBox *bbs = iftReadLabelBoundingBox(mbb_json_path, &bb_labels);
    iftDict *bb_labels_dict = iftCreateDict();
    for (int o = 0; o < bb_labels->n; o++)
        iftInsertIntoDict(bb_labels->val[o], o, bb_labels_dict);
    iftDestroyIntArray(&bb_labels);

    puts("- Getting the Labeled MBBs from GT");
    iftIntArray *gt_labels = iftGetObjectLabels(gt_img);
    iftBoundingBox *gt_mbbs = iftMinLabelsBoundingBox(gt_img, gt_labels, NULL);


    puts("- Computing Scores");
    for (int o = 0; o < gt_labels->n; o++) {
        printf("* Object: %d\n", gt_labels->val[o]);
        printf("gt_mbbs.begin: %d, %d, %d\n", gt_mbbs[o].begin.x, gt_mbbs[o].begin.y, gt_mbbs[o].begin.z);
        printf("gt_mbbs.end: %d, %d, %d\n", gt_mbbs[o].end.x, gt_mbbs[o].end.y, gt_mbbs[o].end.z);
        printf("bbs.begin: %d, %d, %d\n", bbs[o].begin.x, bbs[o].begin.y, bbs[o].begin.z);
        printf("bbs.end: %d, %d, %d\n", bbs[o].end.x, bbs[o].end.y, bbs[o].end.z);

        int bb_obj_idx = iftGetLongValFromDict(gt_labels->val[o], bb_labels_dict);

        float score1 = iftDistanceBetweenMBBCenters(gt_mbbs[o], bbs[bb_obj_idx], gt_img->dx);
        printf("Score 1 = %f mm\n", score1);

        float score2 = iftPercOfLostObjectVoxels(gt_img, bbs[bb_obj_idx], gt_labels->val[o]);
        printf("Score 2 = %f\n", score2);

        float score3 = iftBgVoxelsError(gt_img, bbs[bb_obj_idx], gt_labels->val[o]);
        printf("Score 3 = %f\n\n", score3);
        

        // storing the scores into file
        char out_score_csv[512];
        sprintf(out_score_csv, "%s_%d.csv", out_score_basename, gt_labels->val[o]);
        FILE *score_fp = fopen(out_score_csv, "a+");
        fprintf(score_fp, "%s;%f;%f;%f\n", gt_path, score1, score2, score3);
        fclose(score_fp);
    }
    iftFree(gt_mbbs);


    if (iftDictContainKey("--output-dir", args, NULL)) {
        const char *out_dir = iftGetStrValFromDict("--output-dir", args);
        if (!iftDirExists(out_dir))
            iftMakeDir(out_dir);

        char *test_filename = iftFilename(gt_path, NULL);
        char *out_path = iftJoinPathnames(2, out_dir, test_filename);
    
        puts("- Writing Label Image (Test Label Image + Min Bounding Box Image)");
        puts(out_path);
        iftImage *gt_bin = iftThreshold(gt_img, 1, IFT_INFINITY_INT, 1);
        iftImage *bb_bin = iftCreateImageFromImage(gt_img);
        for (int o = 0; o < gt_labels->n; o++) {
            int bb_obj_idx = iftGetLongValFromDict(gt_labels->val[o], bb_labels_dict);
            iftFillBoundingBoxInImage(bb_bin, bbs[bb_obj_idx], 1);
        }
        iftImage *out_img = iftAdd(gt_bin, bb_bin);
        iftWriteImageByExt(out_img, out_path);
        iftDestroyImage(&gt_bin);
        iftDestroyImage(&bb_bin);
        iftDestroyImage(&out_img);

        iftFree(test_filename);
        iftFree(out_path);
    }

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));


    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(gt_path);
    iftFree(mbb_json_path);
    iftFree(out_score_basename);
    iftDestroyImage(&gt_img);
    iftDestroyDict(&bb_labels_dict);
    iftDestroyIntArray(&gt_labels);
    iftFree(bbs);



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
        "3) Ratio between the number of bg voxels from the estimated BB and the True MBB from Gt";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-t", .long_name = "--ground-truth-label-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Ground Truth (Test Label Image) with the objects of interest."},
        {.short_name = "-i", .long_name = "--input-bb-json", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Input Json Path with the Labeled Bounding Boxes."},
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


void iftGetRequiredArgs(  iftDict *args, char **gt_path, char **mbb_json_path,
                        char **out_score_basename) {
    *gt_path            = iftGetStrValFromDict("--ground-truth-label-image", args);
    *mbb_json_path      = iftGetStrValFromDict("--input-bb-json", args);
    *out_score_basename = iftGetStrValFromDict("--output-score-file-basename", args);

    char *parent_dir = iftParentDir(*out_score_basename);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- Ground Truth Image: \"%s\"\n", *gt_path);
    printf("- Min Bounding Box Json File: \"%s\"\n", *mbb_json_path);
    printf("- Output Score CSV: \"%s\"\n", *out_score_basename);
    puts("-----------------------");
}


float iftDistanceBetweenMBBCenters(iftBoundingBox bb1, iftBoundingBox bb2, float voxel_size) {
    iftVoxel c1 = iftBoundingBoxCenterVoxel(bb1);
    iftVoxel c2 = iftBoundingBoxCenterVoxel(bb2);

    printf("\ngt_obj_mbb_center: %d, %d, %d\n", c1.x, c1.y, c1.z);
    printf("localizer_obj_bb_center: %d, %d, %d\n\n", c2.x, c2.y, c2.z);

    return (iftVoxelDistance(c1, c2) * voxel_size);
}


float iftPercOfLostObjectVoxels(  iftImage *gt_img, iftBoundingBox bb, int label) {

    iftImage *gt_obj_bin = iftThreshold(gt_img, label, label, 1); // extract the object with label <label> and binarize them
    iftImage *bb_obj_bin = iftCreateImageFromImage(gt_img);
    iftFillBoundingBoxInImage(bb_obj_bin, bb, 1);

    iftImage *inv_bb_obj_bin = iftComplement(bb_obj_bin); // gets the complement from the target object's bounding box
    iftImage *lost_obj_img = iftMask(gt_obj_bin, inv_bb_obj_bin); // image with the lost object's voxels
    
    int n_gt_obj_voxels = iftCountObjectSpels(gt_obj_bin, 1); // count the number of voxels from the target object in binary image
    int n_gt_lost_voxels = iftCountObjectSpels(lost_obj_img, 1); // count the number of voxels from the target object in binary image

    float perc_lost_obj_voxels = (n_gt_lost_voxels * 1.0) / n_gt_obj_voxels;

    // printf("n_test_obj_voxels = %d\n", n_test_obj_voxels);
    // printf("n_test_lost_voxels = %d\n", n_test_lost_voxels);
    // printf("perc_lost_obj_voxels = %f\n", perc_lost_obj_voxels);

    // iftWriteImageByExt(gt_obj_bin, "out/gt_obj_bin_%d.hdr", label);
    // iftWriteImageByExt(bb_obj_bin, "out/mbb_obj_bin_%d.hdr", label);
    // iftWriteImageByExt(inv_bb_obj_bin, "out/inv_bb_obj_bin_%d.hdr", label);
    // iftWriteImageByExt(lost_obj_img, "out/lost_obj_img_%d.hdr", label);

    iftDestroyImage(&gt_obj_bin);
    iftDestroyImage(&bb_obj_bin);
    iftDestroyImage(&inv_bb_obj_bin);
    iftDestroyImage(&lost_obj_img);

    return perc_lost_obj_voxels;
}


float iftBgVoxelsError(  iftImage *gt_img, iftBoundingBox bb, int label) {
    iftImage *bb_obj_bin = iftCreateImageFromImage(gt_img);
    iftFillBoundingBoxInImage(bb_obj_bin, bb, 1);

    iftImage *gt_obj_bin = iftThreshold(gt_img, label, label, 1); // extract the object with label <label> and binarize them
    iftImage *inv_gt_obj_bin = iftComplement(gt_obj_bin); // gets the complement from the target object's bounding box

    iftBoundingBox gt_obj_mbb = iftMinBoundingBox(gt_obj_bin, NULL); // object mbb from the Ground Truth
    iftImage *gt_mbb_obj_bin  = iftCreateImageFromImage(gt_obj_bin);
    iftFillBoundingBoxInImage(gt_mbb_obj_bin, gt_obj_mbb, 1);

    // Grount Truth
    iftImage *gt_mbb_bg = iftMask(gt_mbb_obj_bin, inv_gt_obj_bin);
    // iftWriteImageByExt(gt_mbb_bg, "tmp/gt_mbb_bg_%d.hdr", label);
    
    iftImage *bb_bg = iftMask(bb_obj_bin, inv_gt_obj_bin);
    int n_bb_bg_voxels = iftCountObjectSpels(bb_bg, 1); // count the number of voxels from the target object in binary image
    // iftWriteImageByExt(bb_bg, "tmp/bb_bg_%d.hdr", label);
    // printf("n_bb_bg_voxels: %d\n", n_bb_bg_voxels);
    
    iftImage *bg_intersec = iftIntersec(gt_mbb_bg, bb_bg);
    int n_bg_intersec = iftCountObjectSpels(bg_intersec, 1);
    // iftWriteImageByExt(bg_intersec, "tmp/bg_intersec_%d.hdr", label);
    // printf("n_bg_intersec: %d\n", n_bg_intersec);

    float bg_error = (n_bb_bg_voxels - n_bg_intersec) / (n_bb_bg_voxels * 1.0);


    iftDestroyImage(&gt_obj_bin);
    iftDestroyImage(&inv_gt_obj_bin);
    iftDestroyImage(&bb_obj_bin);
    iftDestroyImage(&gt_mbb_obj_bin);

    return bg_error;
}
/*************************************************************/










