#include "ift.h"


/************************** HEADERS **************************/
typedef struct ift_ref_obj_localizer {
    iftIntArray *labels;
    iftIntArray *bb_size;
    iftIntArray *search_region_size;
    iftVoxel *disp_vec;
} iftRefObjLocalizer;


iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **test_img_path, char **test_ref_obj_img_path,
                        char **ref_img_path, char **ref_label_img_path, char **localizer_path,
                        char **out_json_path);
iftRefObjLocalizer *iftReadLocalizerRefObject(const char *json_path);
void iftDestroyRefObjLocalizer(iftRefObjLocalizer **loc);

/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *test_img_path         = NULL;
    char *test_ref_obj_img_path = NULL;
    char *ref_img_path          = NULL;
    char *ref_label_img_path    = NULL;
    char *localizer_path        = NULL;
    char *out_json_path         = NULL;


    iftGetRequiredArgs(args, &test_img_path, &test_ref_obj_img_path, &ref_img_path, &ref_label_img_path, &localizer_path, &out_json_path);

    timer *t1 = iftTic();

    puts("- Reading MRI Test Image");
    iftImage *test_img = iftReadImageByExt(test_img_path);

    puts("- Reading Test Ref Object Image");
    iftImage *test_ref_obj_img = iftReadImageByExt(test_ref_obj_img_path);

    puts("- Reading MRI Reference Image");
    iftImage *ref_img = iftReadImageByExt(ref_img_path);

    puts("- Reading MRI Reference Label Image");
    iftImage *ref_label_img = iftReadImageByExt(ref_label_img_path);

    puts("- Reading Localizer Ref Object");
    iftRefObjLocalizer *loc = iftReadLocalizerRefObject(localizer_path);

    iftLabelBoundingBoxArray *lbbs = iftCreateLabelBoundingBoxArray(loc->labels->n);
    iftCopyIntArray(lbbs->labels->val, loc->labels->val, loc->labels->n);

    puts("- Finding the Best Subcortical Bounding Boxes");
    for (int o = 0; o < loc->labels->n; o++) {
        printf("###### Object: %d\n", loc->labels->val[o]);
        
        iftBoundingBox ref_obj_mbb  = iftMinObjectBoundingBox(test_ref_obj_img, 1, NULL);
        iftVoxel ref_obj_mbb_center = iftBoundingBoxCenterVoxel(ref_obj_mbb);
        iftBoundingBox bb;
        bb.begin.x = bb.begin.y = bb.begin.z = 0;
        bb.end.x = loc->bb_size->val[0] - 1;
        bb.end.y = loc->bb_size->val[1] - 1;
        bb.end.z = loc->bb_size->val[2] - 1;


        iftVoxel test_bb_center = iftVectorSum(ref_obj_mbb_center, loc->disp_vec[o]);
        iftBoundingBox test_bb  = iftCentralizeBoundingBox(bb, test_bb_center);
        printf("test_bb_center: (%d, %d, %d)\n", test_bb_center.x, test_bb_center.y, test_bb_center.z);
        puts("test_bb");
        iftPrintBoundingBox(test_bb);


        iftBoundingBox search_region;
        search_region.begin.x = search_region.begin.y = search_region.begin.z = 0;
        search_region.end.x = loc->search_region_size->val[0] - 1;
        search_region.end.y = loc->search_region_size->val[1] - 1;
        search_region.end.z = loc->search_region_size->val[2] - 1;
        
        iftBoundingBox test_search_region = iftCentralizeBoundingBox(search_region, test_bb_center);
        puts("test_search_region");
        iftPrintBoundingBox(test_search_region);

        if (!iftDictContainKey("--ignore-local-search", args, NULL)) {
            puts("- Running Local Search");
            lbbs->bbs[o] = iftMSPSMaxSubcorticalNMI(test_img, ref_img, ref_label_img, loc->labels->val[o], test_bb, test_search_region);
        }
        else lbbs->bbs[o] = test_bb;
    }

    puts("- Writing the Labeled Subcortical Bounding Boxes");
    iftWriteLabelBoundingBoxArray(lbbs, out_json_path);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(test_img_path);
    iftFree(test_ref_obj_img_path);
    iftFree(ref_img_path);
    iftFree(ref_label_img_path);
    iftFree(localizer_path);
    iftFree(out_json_path);

    iftDestroyImage(&test_img);
    iftDestroyImage(&test_ref_obj_img);
    iftDestroyImage(&ref_img);
    iftDestroyImage(&ref_label_img);
    iftDestroyRefObjLocalizer(&loc);
    iftDestroyLabelBoundingBoxArray(&lbbs);
    

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Find Brain Substructues by placing the learned bounding boxes from a Reference Object.\n" \
        "PS: The test image must be registered in the same space where the localizer was trained";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-t", .long_name = "--test-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="MRI Test Image."},
        {.short_name = "-i", .long_name = "--test-reference-object-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Test Binary Image with the reference object for localization."},
        {.short_name = "-r", .long_name = "--reference-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="MRI Reference Image."},
        {.short_name = "-l", .long_name = "--reference-label-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="MRI Label Reference Image with the Target Objects."},
        {.short_name = "-z", .long_name = "--localizer-ref-object", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Localizer Mean Center."},
        {.short_name = "-o", .long_name = "--output-bb-json", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Label Image with the BBs of the Brain Substructures."},
        {.short_name = "", .long_name = "--ignore-local-search", .has_arg=false,
         .required=false, .help="Flag to ignore the local search."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}



void iftGetRequiredArgs(  iftDict *args, char **test_img_path, char **test_ref_obj_img_path,
                        char **ref_img_path, char **ref_label_img_path, char **localizer_path,
                        char **out_json_path) {
    *test_img_path         = iftGetStrValFromDict("--test-image", args);
    *test_ref_obj_img_path = iftGetStrValFromDict("--test-reference-object-image", args);
    *ref_img_path          = iftGetStrValFromDict("--reference-image", args);
    *ref_label_img_path    = iftGetStrValFromDict("--reference-label-image", args);
    *localizer_path        = iftGetStrValFromDict("--localizer-ref-object", args);
    *out_json_path         = iftGetStrValFromDict("--output-bb-json", args);


    char *parent_dir = iftParentDir(*out_json_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- MRI Test Image: %s\n", *test_img_path);
    printf("- Test Reference Object Image: \"%s\"\n", *test_ref_obj_img_path);
    printf("- MRI Reference Image: %s\n", *ref_img_path);
    printf("- MRI Reference Label Image: %s\n", *ref_label_img_path);
    printf("- Localizer: %s\n", *localizer_path);
    printf("- Output Json Path: %s\n", *out_json_path);
    puts("-----------------------\n");
}



iftRefObjLocalizer *iftReadLocalizerRefObject(const char *json_path) {
    iftRefObjLocalizer *loc = iftAlloc(1, sizeof(iftRefObjLocalizer));

    iftDict *json = iftReadJson(json_path);

    loc->labels = iftGetIntArrayFromDict("labels", json);
    loc->disp_vec = iftAlloc(loc->labels->n, sizeof(iftVoxel));


    for (int o = 0; o < loc->labels->n; o++) {
        char key[512];

        sprintf(key, "data:%d:bounding-box-size", loc->labels->val[o]);
        loc->bb_size = iftGetIntArrayFromDict(key, json);

        sprintf(key, "data:%d:disp-vector", loc->labels->val[o]);
        iftIntArray *disp_vec = iftGetIntArrayFromDict(key, json);
        loc->disp_vec[o].x = disp_vec->val[0];
        loc->disp_vec[o].y = disp_vec->val[1];
        loc->disp_vec[o].z = disp_vec->val[2];

        sprintf(key, "data:%d:search-region-size", loc->labels->val[o]);
        loc->search_region_size = iftGetIntArrayFromDict(key, json);
    }
    iftDestroyDict(&json);
    
    return loc;
}


void iftDestroyRefObjLocalizer(iftRefObjLocalizer **loc) {
    iftRefObjLocalizer *aux = *loc;

    if (aux != NULL) {
        iftDestroyIntArray(&aux->labels);
        iftDestroyIntArray(&aux->search_region_size);
        iftDestroyIntArray(&aux->bb_size);
        iftFree(aux->disp_vec);
        iftFree(aux);
        *loc = NULL;
    }
}
/*************************************************************/












