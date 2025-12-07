#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **test_img_path, char **prob_atlas_path,
                        char **sizes_json_path, char **out_bb_json_path);

#include "iftReadBBSizes.c" // file with the function iftReadBBSizes
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *test_img_path    = NULL;
    char *prob_atlas_path  = NULL;
    char *sizes_json_path  = NULL;
    char *out_bb_json_path = NULL;

    iftGetRequiredArgs(args, &test_img_path, &prob_atlas_path, &sizes_json_path, &out_bb_json_path);

    timer *t1 = iftTic();
    
    puts("- Reading MRI Test Image");
    iftImage *test_img = iftReadImageByExt(test_img_path);
    
    puts("- Read Prob. Atlas");
    iftMObjModel *prob_atlas = iftReadSOSM(prob_atlas_path);
    iftDict *labels_dict_idxs = iftIntArrayToDict(prob_atlas->labels->val, prob_atlas->labels->n);

    puts("- Reading BB Size File");
    iftIntArray *labels = NULL;
    iftIntMatrix *size_mat = iftReadBBSizes(sizes_json_path, &labels);

    puts("- Finding BBs");
    iftLabelBoundingBoxArray *lbbs = iftCreateLabelBoundingBoxArray(labels->n);
    iftCopyIntArray(lbbs->labels->val, labels->val, lbbs->labels->n);

    iftImage *seg_img = iftSegmentBySOSM_S(test_img, prob_atlas, 0, 0);
    iftBoundingBox *seg_img_bbs = iftMinLabelsBoundingBox(seg_img, labels, NULL);

    for (int o = 0; o < labels->n; o++) {
        printf("\nObject: %d\n", labels->val[o]);

        iftBoundingBox size_bb;
        size_bb.begin.x = size_bb.begin.y = size_bb.begin.z = 0;
        size_bb.end.x   = iftMatrixElem(size_mat, 0, o) - 1;
        size_bb.end.y   = iftMatrixElem(size_mat, 1, o) - 1;
        size_bb.end.z   = iftMatrixElem(size_mat, 2, o) - 1;

        iftVoxel seg_img_bb_center = iftBoundingBoxCenterVoxel(seg_img_bbs[o]);

        lbbs->bbs[o] = iftCentralizeBoundingBox(size_bb, seg_img_bb_center);
        puts("seg_img_bbs");
        iftPrintBoundingBox(seg_img_bbs[o]);
        printf("seg_img_bb_center: (%d, %d, %d)\n", seg_img_bb_center.x, seg_img_bb_center.y, seg_img_bb_center.z);
        puts("resulting bb");
        iftPrintBoundingBox(lbbs->bbs[o]);
    }


    puts("- Writing Output Resized Label Bounding Boxes");
    iftWriteLabelBoundingBoxArray(lbbs, out_bb_json_path);


    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyImage(&test_img);
    iftDestroyMObjModel(&prob_atlas);
    iftDestroyIntArray(&labels);
    iftDestroyIntMatrix(&size_mat);
    iftDestroyDict(&labels_dict_idxs);


    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Finds the Label Bounding Boxes by segmenting the target objects with a prob atlas.\n" \
        "- The bounding boxes drawn by the user are centralized in the center of the bounding boxes from the " \
        "resulting segmentations.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-t", .long_name = "--test-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="MRI Test Image."},
        {.short_name = "-m", .long_name = "--prob-atlas", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Prob. Atlas from the Target Objects."},
        {.short_name = "-r", .long_name = "--sizes-json", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Json File with the Sizes for each Labeled Bounding Boxes."},
        {.short_name = "-o", .long_name = "--output-bb-json", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Json File with the Label BBs."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **test_img_path, char **prob_atlas_path,
                        char **sizes_json_path, char **out_bb_json_path) {
    *test_img_path    = iftGetStrValFromDict("--test-image", args);
    *prob_atlas_path  = iftGetStrValFromDict("--prob-atlas", args);
    *sizes_json_path  = iftGetStrValFromDict("--sizes-json", args);
    *out_bb_json_path = iftGetStrValFromDict("--output-bb-json", args);

    char *parent_dir = iftParentDir(*out_bb_json_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- MRI Test Image: %s\n", *test_img_path);
    printf("- Prob. Atlas of The Target Objects: \"%s\"\n", *prob_atlas_path);
    printf("- Json File with the BB sizes: \"%s\"\n", *sizes_json_path);
    printf("- Output Resized Bounding Box Json File: \"%s\"\n", *out_bb_json_path);
    puts("-----------------------\n");
}
/*************************************************************/


