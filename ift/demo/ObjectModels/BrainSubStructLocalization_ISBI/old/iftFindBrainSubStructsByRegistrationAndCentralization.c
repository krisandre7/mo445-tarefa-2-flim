#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **test_label_img_path, char **ref_bbs_path,
                        char **out_bbs_path);
void iftGetOptionalArgs(  iftDict *args, double *alpha);
iftBoundingBox *iftFindROIs(  iftImage *test_label_img,   iftBoundingBox *ref_bbs,
                              iftIntArray *bb_labels, double alpha, iftIntArray **out_labels);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory arguments
    char *test_label_img_path = NULL;
    char *ref_bbs_path = NULL;
    char *out_bbs_path       = NULL;
    // optional arguments
    double alpha;

    iftGetRequiredArgs(args, &test_label_img_path, &ref_bbs_path, &out_bbs_path);
    iftGetOptionalArgs(args, &alpha);

    timer *t1 = iftTic();

    puts("- Reading Test Label Image");
    iftImage *test_label_img = iftReadImageByExt(test_label_img_path);

    puts("- Reading Ref. Label Image");
    iftIntArray *bb_labels = NULL;
    iftBoundingBox *ref_bbs = iftReadLabelBoundingBox(ref_bbs_path, &bb_labels);

    printf(" - Finding ROIs\n");
    iftIntArray *out_labels = NULL;
    iftBoundingBox *out_bbs = iftFindROIs(test_label_img, ref_bbs, bb_labels, alpha, &out_labels);

    printf(" - Writing ROIs\n");
    iftWriteLabelBoundingBox(out_bbs, out_labels, out_bbs_path);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(test_label_img_path);
    iftFree(ref_bbs_path);
    iftFree(out_bbs_path);
    iftDestroyImage(&test_label_img);
    iftDestroyIntArray(&bb_labels);
    iftDestroyIntArray(&out_labels);
    iftFree(ref_bbs);
    iftFree(out_bbs);

    return 0;
}



/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Find the Regions of Interest of brain substructures from Reference Image with the " \
        "required objects. The ROIs are the bounding boxes of all required objects in the Reference Image.\n" \
        "Such ROIs are centralized on MBBs from the Test Image.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-t", .long_name = "--test-label-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Test Label Image with required target objects of interest."},
        {.short_name = "-r", .long_name = "--reference-image-bb-json", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Json Path with the Labeled Bounding Boxes from the Reference Image."},
        {.short_name = "-o", .long_name = "--output-bb-json", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Centralized Label Bounding Boxes."},
        {.short_name = "-a", .long_name = "--alpha", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Factor to scale the found regions of interest (Bounding Boxes). Default: 1.0"}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **test_label_img_path, char **ref_bbs_path,
                        char **out_bbs_path) {
    *test_label_img_path = iftGetStrValFromDict("--test-label-image", args);
    *ref_bbs_path        = iftGetStrValFromDict("--reference-image-bb-json", args);
    *out_bbs_path        = iftGetStrValFromDict("--output-bb-json", args);

    char *parent_dir = iftParentDir(*out_bbs_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- Test Label Image: \"%s\"\n", *test_label_img_path);
    printf("- Reference Labeled Bounding Boxes: \"%s\"\n", *ref_bbs_path);
    printf("- Output Bounding Boxes: \"%s\"\n", *out_bbs_path);
    puts("-----------------------");
}


void iftGetOptionalArgs(  iftDict *args, double *alpha) {
    if (iftDictContainKey("--alpha", args, NULL))
        *alpha = iftGetDblValFromDict("--alpha", args);
    else *alpha = 1.0;

    if (*alpha <= 0.0)
        iftError("Invalid Scaling Factor: %lf <= 0.0", "iftGetOptionalArgs", *alpha);
    
    printf("- Scaling Factor: %lf\n", *alpha);
    puts("-----------------------\n");
}


iftBoundingBox *iftFindROIs(  iftImage *test_label_img,   iftBoundingBox *ref_bbs,
                              iftIntArray *bb_labels, double alpha, iftIntArray **out_labels) {
    iftIntArray *test_labels  = iftGetObjectLabels(test_label_img);
    iftBoundingBox *test_mbbs = iftMinLabelsBoundingBox(test_label_img, test_labels, NULL);
    iftBoundingBox *out_bbs   = iftAlloc(test_labels->n, sizeof(iftBoundingBox));


    iftDict *bb_labels_dict = iftCreateDict();
    for (int o = 0; o < bb_labels->n; o++)
        iftInsertIntoDict(bb_labels->val[o], o, bb_labels_dict);
    

    for (int o = 0; o < test_labels->n; o++) {
        printf("\nObject: %d\n", test_labels->val[o]);
        int bb_obj_idx = iftGetLongValFromDict(test_labels->val[o], bb_labels_dict);

        iftBoundingBox ref_obj_bb  = iftScaleBoundingBox(ref_bbs[bb_obj_idx], alpha);
        iftVoxel test_mbb_center   = iftBoundingBoxCenterVoxel(test_mbbs[o]);
        iftVoxel ref_obj_bb_center = iftBoundingBoxCenterVoxel(ref_obj_bb);
        iftVoxel disp              = iftVectorSub(test_mbb_center, ref_obj_bb_center);
        printf("test_mbbs.begin: (%d, %d, %d)\n", test_mbbs[o].begin.x, test_mbbs[o].begin.y, test_mbbs[o].begin.z);
        printf("test_mbbs.end: (%d, %d, %d)\n", test_mbbs[o].end.x, test_mbbs[o].end.y, test_mbbs[o].end.z);
        printf("test_mbb_center: (%d, %d, %d)\n\n", test_mbb_center.x, test_mbb_center.y, test_mbb_center.z);

        printf("ref_obj_bb.begin: (%d, %d, %d)\n", ref_obj_bb.begin.x, ref_obj_bb.begin.y, ref_obj_bb.begin.z);
        printf("ref_obj_bb.end: (%d, %d, %d)\n", ref_obj_bb.end.x, ref_obj_bb.end.y, ref_obj_bb.end.z);
        printf("ref_obj_bb_center: (%d, %d, %d)\n", ref_obj_bb_center.x, ref_obj_bb_center.y, ref_obj_bb_center.z);
        printf("disp: (%d, %d, %d)\n\n", disp.x, disp.y, disp.z);

        // centralize reference bounding box on the test mbb
        out_bbs[o].begin.x = ref_obj_bb.begin.x + disp.x;
        out_bbs[o].begin.y = ref_obj_bb.begin.y + disp.y;
        out_bbs[o].begin.z = ref_obj_bb.begin.z + disp.z;
        out_bbs[o].end.x   = ref_obj_bb.end.x + disp.x;
        out_bbs[o].end.y   = ref_obj_bb.end.y + disp.y;
        out_bbs[o].end.z   = ref_obj_bb.end.z + disp.z;

        iftVoxel out_bbs_center = iftBoundingBoxCenterVoxel(out_bbs[o]);
        printf("out_bbs.begin: (%d, %d, %d)\n", out_bbs[o].begin.x, out_bbs[o].begin.y, out_bbs[o].begin.z);
        printf("out_bbs.end: (%d, %d, %d)\n", out_bbs[o].end.x, out_bbs[o].end.y, out_bbs[o].end.z);
        printf("centralized out_bbs_center: (%d, %d, %d)\n\n", out_bbs_center.x, out_bbs_center.y, out_bbs_center.z);
    }
    iftDestroyDict(&bb_labels_dict);

    *out_labels = test_labels;

    return out_bbs;
}
/*************************************************************/


