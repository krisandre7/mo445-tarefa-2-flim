#include "ift.h"


/************************** HEADERS **************************/
typedef struct ift_ref_obj_localizer {
    iftIntArray *labels;
    iftImageDomain *roi_sizes;
    iftVoxel *disp_vec;
} iftRefObjLocalizer;


iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **test_ref_obj_img_path, char **localizer_path,
                        char **out_bbs_path);
iftRefObjLocalizer *iftReadRefObjLocalizer(const char *localizer_path);
void iftDestroyRefObjLocalizer(iftRefObjLocalizer **localizer);
iftBoundingBox *iftFindBrainSubStructsByRefObject(  iftImage *test_ref_obj_img,   iftRefObjLocalizer *localizer);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *test_ref_obj_img_path = NULL;
    char *localizer_path        = NULL;
    char *out_bbs_path          = NULL;

    iftGetRequiredArgs(args, &test_ref_obj_img_path, &localizer_path, &out_bbs_path);

    timer *t1 = iftTic();

    puts("- Reading Test Ref. Object Image");
    iftImage *test_ref_obj_img = iftReadImageByExt(test_ref_obj_img_path);
    
    puts("- Reading Localizer");
    iftRefObjLocalizer *localizer = iftReadRefObjLocalizer(localizer_path);

    puts("- Finding ROIs");
    iftBoundingBox *bbs = iftFindBrainSubStructsByRefObject(test_ref_obj_img, localizer);

    puts("- Writing ROIs");
    iftWriteLabelBoundingBox(bbs, localizer->labels, out_bbs_path);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(test_ref_obj_img_path);
    iftFree(localizer_path);
    iftFree(out_bbs_path);
    iftDestroyRefObjLocalizer(&localizer);
    iftDestroyImage(&test_ref_obj_img);
    iftFree(bbs);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Find Brain Substructues by placing ROIs from a Reference Object.\n" \
        "PS: The test image must be registered in the same space where the localizer was trained";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-l", .long_name = "--test-reference-object-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Test Binary Image with the reference object for localization."},
        {.short_name = "-z", .long_name = "--localizer", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Json file with the Localizer."},
        {.short_name = "-o", .long_name = "--output-bb-json", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Label Image with the ROIs of the Brain Substructues."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **test_ref_obj_img_path, char **localizer_path,
                        char **out_bbs_path) {
    *test_ref_obj_img_path = iftGetStrValFromDict("--test-reference-object-image", args);
    *localizer_path        = iftGetStrValFromDict("--localizer", args);
    *out_bbs_path      = iftGetStrValFromDict("--output-bb-json", args);

    char *parent_dir = iftParentDir(*out_bbs_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);


    puts("-----------------------");
    printf("- Reference Object Image: \"%s\"\n", *test_ref_obj_img_path);
    printf("- Localizer: \"%s\"\n", *localizer_path);
    printf("- Output ROIs Json: \"%s\"\n", *out_bbs_path);
    puts("-----------------------\n");
}


iftRefObjLocalizer *iftReadRefObjLocalizer(const char *localizer_path) {
    iftRefObjLocalizer *localizer = iftAlloc(1, sizeof(iftRefObjLocalizer));

    iftJson *json = iftReadJson(localizer_path);

    localizer->labels    = iftGetJIntArray(json, "labels");
    localizer->roi_sizes = iftAlloc(localizer->labels->n, sizeof(iftImageDomain));
    localizer->disp_vec  = iftAlloc(localizer->labels->n, sizeof(iftVoxel));

    for (int o = 0; o < localizer->labels->n; o++) {
        char key[128];

        sprintf(key, "localizer:%d:ROI-sizes", localizer->labels->val[o]);
        iftIntArray *ROI_sizes = iftGetJIntArray(json, key);
        localizer->roi_sizes[o].xsize = ROI_sizes->val[0];
        localizer->roi_sizes[o].ysize = ROI_sizes->val[1];
        localizer->roi_sizes[o].zsize = ROI_sizes->val[2];
        iftDestroyIntArray(&ROI_sizes);
        // printf("localizer->roi_sizes[%d]: (%d, %d, %d)\n", o, localizer->roi_sizes[o].xsize, localizer->roi_sizes[o].ysize, localizer->roi_sizes[o].zsize);


        sprintf(key, "localizer:%d:displacement-vector", localizer->labels->val[o]);
        iftIntArray *disp_vecs_arr = iftGetJIntArray(json, key);
        localizer->disp_vec[o].x   = disp_vecs_arr->val[0];
        localizer->disp_vec[o].y   = disp_vecs_arr->val[1];
        localizer->disp_vec[o].z   = disp_vecs_arr->val[2];
        iftDestroyIntArray(&disp_vecs_arr);

        // printf("localizer->disp_vec[%d]: (%d, %d, %d)\n", o, localizer->disp_vec[o].x, localizer->disp_vec[o].y, localizer->disp_vec[o].z);

    }
    iftDestroyJson(&json);


    return localizer;
}


void iftDestroyRefObjLocalizer(iftRefObjLocalizer **localizer) {
    iftRefObjLocalizer *aux = *localizer;

    if (aux != NULL) {
        iftDestroyIntArray(&aux->labels);
        iftFree(aux->disp_vec);
        iftFree(aux->roi_sizes);
        iftFree(aux);
        *localizer = NULL;
    }
}

iftBoundingBox *iftFindBrainSubStructsByRefObject(  iftImage *test_ref_obj_img,   iftRefObjLocalizer *localizer) {
    iftBoundingBox *bbs = iftAlloc(localizer->labels->n, sizeof(iftBoundingBox));

    iftBoundingBox test_mbb  = iftMinBoundingBox(test_ref_obj_img, NULL); // Ref. Object Image must be binary (only one object)
    iftVoxel test_mbb_center = iftBoundingBoxCenterVoxel(test_mbb);

    for (int o = 0; o < localizer->labels->n; o++) {
        printf("Object: %d\n", localizer->labels->val[o]);
        
        iftVoxel target_obj_mbb_center = iftVectorSum(test_mbb_center, localizer->disp_vec[o]);
        printf("test_mbb_center: (%d, %d, %d)\n", test_mbb_center.x, test_mbb_center.y, test_mbb_center.z);
        printf("disp_vector: (%d, %d, %d)\n", localizer->disp_vec[o].x, localizer->disp_vec[o].y, localizer->disp_vec[o].z);
        printf("resulting: target_obj_mbb_center: (%d, %d, %d)\n\n", target_obj_mbb_center.x, target_obj_mbb_center.y, target_obj_mbb_center.z);

        int half_xsize = iftRound(localizer->roi_sizes[o].xsize / 2.0);
        int half_ysize = iftRound(localizer->roi_sizes[o].ysize / 2.0);
        int half_zsize = iftRound(localizer->roi_sizes[o].zsize / 2.0);

        bbs[o].begin.x = target_obj_mbb_center.x - half_xsize;
        bbs[o].begin.y = target_obj_mbb_center.y - half_ysize;
        bbs[o].begin.z = target_obj_mbb_center.z - half_zsize;
        bbs[o].end.x   = target_obj_mbb_center.x + half_xsize;
        bbs[o].end.y   = target_obj_mbb_center.y + half_ysize;
        bbs[o].end.z   = target_obj_mbb_center.z + half_zsize;
        printf("MBB.begin: (%d, %d, %d)\n", bbs[o].begin.x, bbs[o].begin.y, bbs[o].begin.z);
        printf("MBB.end: (%d, %d, %d)\n\n", bbs[o].end.x, bbs[o].end.y, bbs[o].end.z);
    }


    return bbs;
}
/*************************************************************/


