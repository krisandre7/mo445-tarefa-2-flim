#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **ref_label_img_path, char **ref_bin_obj_img_path,
                        char **out_json_path, char **out_rois_path);
void iftGetOptionalArgs(  iftDict *args, float *alpha);
iftBoundingBox *iftFindLabelMinBoundingBoxes(  iftImage *ref_label_img,   iftIntArray *labels, float alpha);
iftBoundingBox iftEnlargeBoundingBox(  iftBoundingBox bb, float alpha);
iftVoxel *iftFindDispVectorBetweenBoundingBoxCenters(  iftImage *ref_label_img,   iftImage *ref_bin_obj_img,
                                                       iftIntArray *labels);
void iftWriteLocalizer(  iftBoundingBox *largest_mbbs,   iftVoxel *disp_vecs,   iftIntArray *labels,
                      const char *out_json_path);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *ref_label_img_path   = NULL;
    char *ref_bin_obj_img_path = NULL;
    char *out_json_path        = NULL;
    char *out_rois_path        = NULL;
    // optional args
    float alpha = 1.1;
    
    iftRandomSeed(time(NULL));

    iftGetRequiredArgs(args, &ref_label_img_path, &ref_bin_obj_img_path, &out_json_path, &out_rois_path);
    iftGetOptionalArgs(args, &alpha);

    timer *t1 = iftTic();

    puts("- Reading Ref. Label Image");
    iftImage *ref_label_img = iftReadImageByExt(ref_label_img_path);

    puts("- Reading Ref. Object Image");
    iftImage *ref_bin_obj_img = iftReadImageByExt(ref_bin_obj_img_path);

    iftIntArray *labels = iftGetObjectLabels(ref_label_img);

    puts("- Finding enlarged Min. Bounding Boxes");
    iftBoundingBox *embbs = iftFindLabelMinBoundingBoxes(ref_label_img, labels, alpha);

    puts("\n- Writing the ROI image");
    iftImage *rois_img = iftCreateImageFromImage(ref_label_img);
    for (int o = 0; o < labels->n; o++) {
        iftFillBoundingBoxInImage(rois_img, embbs[o], labels->val[o]);
    }
    iftWriteImageByExt(rois_img, out_rois_path);

    puts("\n- Finding the Displacement Vectors");
    iftVoxel *disp_vecs = iftFindDispVectorBetweenBoundingBoxCenters(ref_label_img, ref_bin_obj_img, labels);

    puts("\n- Saving the Output Localizer (Json)");
    iftWriteLocalizer(embbs, disp_vecs, labels, out_json_path);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(ref_label_img_path);
    iftFree(ref_bin_obj_img_path);
    iftFree(out_json_path);
    iftFree(out_rois_path);
    iftDestroyIntArray(&labels);
    iftFree(embbs);
    iftDestroyImage(&ref_label_img);
    iftDestroyImage(&ref_bin_obj_img);
    iftFree(disp_vecs);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- This program builds a brain substructure localizer from a reference label image, " \
        "and its binary image with the reference object.\n" \
        "- In order to simulate the real action of a user, when finding the min. bounding box (mbb) of each " \
        "required object, a small movement (displacement) is applied into the mbb.\n" \
        "- The size of the regions of interest will be the min. bounding boxes from the target objects " \
        "enlarged by a factor of alpha (optional).\n" \
        "- The position of the regions is the displacement vector between the center of a reference object and " \
        "the centers of the min. bounding boxes from a reference image (standard space)";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-l", .long_name = "--reference-label-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Reference Image with the target objects of interest."},
        {.short_name = "-b", .long_name = "--reference-object-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Reference Binary Image with the reference object for localization."},
        {.short_name = "-o", .long_name = "--output-localizer", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output json file that stores the localizer information."},
        {.short_name = "-r", .long_name = "--output-rois-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output label image with the ROIs (min. bounding boxes) computed."},
        {.short_name = "-a", .long_name = "--alpha", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Factor to scale the found regions of interest (Bounding Boxes). Default: 1.1"}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **ref_label_img_path, char **ref_bin_obj_img_path,
                        char **out_json_path, char **out_rois_path) {
    *ref_label_img_path   = iftGetStrValFromDict("--reference-label-image", args);
    *ref_bin_obj_img_path = iftGetStrValFromDict("--reference-object-image", args);
    *out_json_path        = iftGetStrValFromDict("--output-localizer", args);
    *out_rois_path        = iftGetStrValFromDict("--output-rois-image", args);

    char *parent_dir = iftParentDir(*out_json_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);
    parent_dir = iftParentDir(*out_rois_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- Reference Label Image: \"%s\"\n", *ref_label_img_path);
    printf("- Reference Object Image: \"%s\"\n", *ref_bin_obj_img_path);
    printf("- Output (json) Localizer: \"%s\"\n", *out_json_path);
    printf("- Output ROI Image: \"%s\"\n", *out_rois_path);
    puts("-----------------------");
}


void iftGetOptionalArgs(  iftDict *args, float *alpha) {
    if (iftDictContainKey("--alpha", args, NULL))
        *alpha = iftGetDblValFromDict("--alpha", args);
    else *alpha = 1.1;

    if (*alpha < 0.0)
        iftError("Alpha factor %d < 0.0", "iftGetOptionalArgs", *alpha);

    printf("- Scale Factor: %lf\n", *alpha);
    puts("-----------------------\n");
}


iftBoundingBox *iftFindLabelMinBoundingBoxes(  iftImage *ref_label_img,   iftIntArray *labels, float alpha) {
    iftBoundingBox *embbs          = iftMinLabelsBoundingBox(ref_label_img, labels, NULL);

    iftRandomSeed(time(NULL));

    for (int o = 0; o < labels->n; o++) {
        printf("\nObject: %d\n", labels->val[o]);

        printf("eMBB.begin: (%d, %d, %d)\n", embbs[o].begin.x, embbs[o].begin.y, embbs[o].begin.z);
        printf("eMBB.end: (%d, %d, %d)\n\n", embbs[o].end.x, embbs[o].end.y, embbs[o].end.z);

        // Since it is too difficult that the user draws a mbb centralized in the target object,
        // we try to simulate a real situation by enlarging the mbbs by a factor and moving the mbb's center
        // by a random displacement vector.
        embbs[o] = iftEnlargeBoundingBox(embbs[o], alpha);
        iftVector disp_mbb_vec;
        disp_mbb_vec.x = iftRandomInteger(-3, 3);
        disp_mbb_vec.y = iftRandomInteger(-3, 3);
        disp_mbb_vec.z = iftRandomInteger(-3, 3);

        printf("eMBB.begin: (%d, %d, %d)\n", embbs[o].begin.x, embbs[o].begin.y, embbs[o].begin.z);
        printf("eMBB.end: (%d, %d, %d)\n\n", embbs[o].end.x, embbs[o].end.y, embbs[o].end.z);
        printf("disp_mbb_vec: (%.0f, %.0f, %.0f)\n\n", disp_mbb_vec.x, disp_mbb_vec.y, disp_mbb_vec.z);

        embbs[o].begin.x += disp_mbb_vec.x;
        embbs[o].end.x   += disp_mbb_vec.x;
        embbs[o].begin.y += disp_mbb_vec.y;
        embbs[o].end.y   += disp_mbb_vec.y;
        embbs[o].begin.z += disp_mbb_vec.z;
        embbs[o].end.z   += disp_mbb_vec.z;

        printf("moved eMBB.begin: (%d, %d, %d)\n", embbs[o].begin.x, embbs[o].begin.y, embbs[o].begin.z);
        printf("moved eMBB.end: (%d, %d, %d)\n\n", embbs[o].end.x, embbs[o].end.y, embbs[o].end.z);
    }

    return embbs;
}


iftBoundingBox iftEnlargeBoundingBox(  iftBoundingBox bb, float alpha) {
    iftMatrix *S = iftScaleMatrix(alpha, alpha, alpha);

    iftPoint begin = {.x = bb.begin.x, .y = bb.begin.y, .z = bb.begin.z};
    iftPoint end   = {.x = bb.end.x,   .y = bb.end.y,   .z = bb.end.z};

    begin = iftTransformPoint(S, begin);
    end   = iftTransformPoint(S, end);
    iftBoundingBox ebb;
    ebb.begin.x = begin.x;
    ebb.begin.y = begin.y;
    ebb.begin.z = begin.z;
    ebb.end.x   = end.x;
    ebb.end.y   = end.y;
    ebb.end.z   = end.z;

    iftDestroyMatrix(&S);

    if (ebb.begin.x < 0) {
        printf("* ebb.begin.x: %d < 0... it is gonna be 0\n", ebb.begin.x);
        ebb.begin.x = 0;
    }
    if (ebb.begin.y < 0) {
        printf("* ebb.begin.y: %d < 0... it is gonna be 0\n", ebb.begin.y);
        ebb.begin.y = 0;
    }
    if (ebb.begin.z < 0) {
        printf("* ebb.begin.z: %d < 0... it is gonna be 0\n", ebb.begin.z);
        ebb.begin.z = 0;
    }

    return ebb;
}


iftVoxel *iftFindDispVectorBetweenBoundingBoxCenters(  iftImage *ref_label_img,   iftImage *ref_bin_obj_img,
                                                       iftIntArray *labels) {
    iftBoundingBox *mbbs        = iftMinLabelsBoundingBox(ref_label_img, labels, NULL);

    iftBoundingBox mbb_ref_obj  = iftMinBoundingBox(ref_bin_obj_img, NULL); // Ref. Object Image must be binary
    iftVoxel mbb_ref_obj_center = iftBoundingBoxCenterVoxel(mbb_ref_obj);
    // iftImage *mbb_ref_obj_img = iftCreateImageFromImage(ref_bin_obj_img);
    // iftFillBoundingBoxInImage(mbb_ref_obj_img, mbb_ref_obj, 1);
    // iftWriteImageByExt(mbb_ref_obj_img, "tmp/mbb_ref_obj_img.scn");
    // iftImage *mbbs_img = iftCreateImageFromImage(ref_bin_obj_img);


    iftVoxel *disp_vecs = iftAlloc(labels->n, sizeof(iftVoxel));

    for (int o = 0; o < labels->n; o++) {
        // iftFillBoundingBoxInImage(mbbs_img, mbbs[o], labels->val[o]);
        // iftWriteImageByExt(mbbs_img, "tmp/mbbs_img_%d.scn", labels->val[o]);
        
        iftVoxel mbb_center = iftBoundingBoxCenterVoxel(mbbs[o]);

        iftVoxel disp  = iftVectorSub(mbb_center, mbb_ref_obj_center);
        disp_vecs[o].x = disp.x;
        disp_vecs[o].y = disp.y;
        disp_vecs[o].z = disp.z;

        printf("Object: %d\n", labels->val[o]);
        printf("- Disp Vector: (%d, %d, %d)\n", disp_vecs[o].x, disp_vecs[o].y, disp_vecs[o].z);
    }
    iftFree(mbbs);


    return disp_vecs;
}


void iftWriteLocalizer(  iftBoundingBox *largest_mbbs,   iftVoxel *disp_vecs,   iftIntArray *labels,
                      const char *out_json_path) {
    iftJson *json = iftCreateJsonRoot();

    // saving labels
    iftAddIntArrayToJson(json, "labels", labels);

    iftAddJDictToJson(json, "localizer", iftCreateJDict());

    for (int o = 0; o < labels->n; o++) {
        char dict_key[128];
        sprintf(dict_key, "localizer:%d", labels->val[o]);
        iftAddJDictToJson(json, dict_key, iftCreateJDict());
        
        char key[128];
        sprintf(key, "%s:ROI-sizes", dict_key);
        iftIntArray *ROI_sizes = iftCreateIntArray(3);
        ROI_sizes->val[0] = largest_mbbs[o].end.x - largest_mbbs[o].begin.x + 1;
        ROI_sizes->val[1] = largest_mbbs[o].end.y - largest_mbbs[o].begin.y + 1;
        ROI_sizes->val[2] = largest_mbbs[o].end.z - largest_mbbs[o].begin.z + 1;
        iftAddIntArrayToJson(json, key, ROI_sizes);
        iftDestroyIntArray(&ROI_sizes);

        sprintf(key, "%s:displacement-vector", dict_key);
        iftIntArray *disp_vecs_arr = iftCreateIntArray(3);
        disp_vecs_arr->val[0] = disp_vecs[o].x;
        disp_vecs_arr->val[1] = disp_vecs[o].y;
        disp_vecs_arr->val[2] = disp_vecs[o].z;
        iftAddIntArrayToJson(json, key, disp_vecs_arr);
        iftDestroyIntArray(&disp_vecs_arr);
    }

    iftWriteJson(json, out_json_path);
    iftDestroyJson(&json);
}
/*************************************************************/


