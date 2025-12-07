#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **ref_label_img_path, char **out_roi_path);
void iftGetOptionalArgs(  iftDict *args, double *alpha);
iftImage *iftFindROIs(  iftImage *ref_label_img, double alpha);
iftBoundingBox iftSimulateUserBoundingBox(iftBoundingBox bb, double alpha);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory arguments
    char *ref_label_img_path = NULL;
    char *out_roi_path       = NULL;
    // optional arguments
    double alpha;

    iftRandomSeed(time(NULL));
    
    iftGetRequiredArgs(args, &ref_label_img_path, &out_roi_path);
    iftGetOptionalArgs(args, &alpha);

    timer *t1 = iftTic();

    puts("- Reading Ref. Label Image");
    iftImage *ref_label_img = iftReadImageByExt(ref_label_img_path);

    printf(" - Finding ROIs\n");
    iftImage *rois = iftFindROIs(ref_label_img, alpha);

    printf(" - Writing ROIs\n");
    iftWriteImageByExt(rois, out_roi_path);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(ref_label_img_path);
    iftFree(out_roi_path);
    iftDestroyImage(&ref_label_img);
    iftDestroyImage(&rois);

    return 0;
}



/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Find the Regions of Interest of brain substructures from reference image with the " \
        "required objects. The ROIs are the bounding boxes of all required objects.\n" \
        "- In order to better simulate a user action, the Min Bounding Boxes (MBBs) are scaled by a small factor and " \
        "a random translation is applied, since it is too hard that the user draws " \
        "perfect MBBs centralized on the target objects.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-r", .long_name = "--reference-label-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Reference Label Image with required target objects of interest."},
        {.short_name = "-o", .long_name = "--output-rois", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Label Image with the computed ROIs."},
        {.short_name = "-a", .long_name = "--alpha", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Factor to scale the found regions of interest (Bounding Boxes). Default: 1.2"}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **ref_label_img_path, char **out_roi_path) {
    *ref_label_img_path         = iftGetStrValFromDict("--reference-label-image", args);
    *out_roi_path         = iftGetStrValFromDict("--output-rois", args);

    char *parent_dir = iftParentDir(*out_roi_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- Reference Label Image: \"%s\"\n", *ref_label_img_path);
    printf("- Output Localizer: \"%s\"\n", *out_roi_path);
    puts("-----------------------");
}


void iftGetOptionalArgs(  iftDict *args, double *alpha) {
    if (iftDictContainKey("--alpha", args, NULL))
        *alpha = iftGetDblValFromDict("--alpha", args);
    else *alpha = 1.2;

    if (*alpha <= 0.0)
        iftError("Invalid Scaling Factor: %lf <= 0.0", "iftGetOptionalArgs", *alpha);
    
    printf("- Scaling Factor: %lf\n", *alpha);
    puts("-----------------------\n");
}



iftImage *iftFindROIs(  iftImage *ref_label_img, double alpha) {
    iftIntArray *labels  = iftGetObjectLabels(ref_label_img);
    iftBoundingBox *mbbs = iftMinLabelsBoundingBox(ref_label_img, labels, NULL);

    iftImage *rois = iftCreateImageFromImage(ref_label_img);

    for (int o = 0; o < labels->n; o++) {
        printf("\nObject: %d\n", labels->val[o]);
        mbbs[o] = iftSimulateUserBoundingBox(mbbs[o], alpha);
        iftFillBoundingBoxInImage(rois, mbbs[o], labels->val[o]);
    }

    iftDestroyIntArray(&labels);
    iftFree(mbbs);

    return rois;
}


iftBoundingBox iftSimulateUserBoundingBox(iftBoundingBox bb, double alpha) {
    printf("bb.begin: (%d, %d, %d)\n", bb.begin.x, bb.begin.y, bb.begin.z);
    printf("bb.end: (%d, %d, %d)\n\n", bb.end.x, bb.end.y, bb.end.z);

    iftBoundingBox ubb = iftScaleBoundingBox(bb, alpha);

    printf("ubb.begin: (%d, %d, %d)\n", ubb.begin.x, ubb.begin.y, ubb.begin.z);
    printf("ubb.end: (%d, %d, %d)\n\n", ubb.end.x, ubb.end.y, ubb.end.z);

    // Since it is too difficult that the user draws a mbb centralized in the target object,
    // we try to simulate a real situation by enlarging the mbbs by a factor and moving the mbb's center
    // by a random displacement vector.
    iftVector disp_mbb_vec;
    disp_mbb_vec.x = iftRandomInteger(-3, 3);
    disp_mbb_vec.y = iftRandomInteger(-3, 3);
    disp_mbb_vec.z = iftRandomInteger(-3, 3);

    ubb.begin.x += disp_mbb_vec.x;
    ubb.end.x   += disp_mbb_vec.x;
    ubb.begin.y += disp_mbb_vec.y;
    ubb.end.y   += disp_mbb_vec.y;
    ubb.begin.z += disp_mbb_vec.z;
    ubb.end.z   += disp_mbb_vec.z;

    printf("disp_mbb_vec: (%.0f, %.0f, %.0f)\n\n", disp_mbb_vec.x, disp_mbb_vec.y, disp_mbb_vec.z);
    printf("ubb.begin: (%d, %d, %d)\n", ubb.begin.x, ubb.begin.y, ubb.begin.z);
    printf("ubb.end: (%d, %d, %d)\n\n", ubb.end.x, ubb.end.y, ubb.end.z);

    return ubb;
}
/*************************************************************/


