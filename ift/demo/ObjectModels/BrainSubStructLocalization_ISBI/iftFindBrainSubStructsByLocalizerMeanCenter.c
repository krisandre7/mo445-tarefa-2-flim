#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **test_img_path, char **ref_img_path, char **ref_label_img_path,
                        char **localizer_path, char **out_json_path);
iftLabelBoundingBoxArray *iftReadLocalizerMeanCenter(const char *json_path, iftBoundingBox **out_search_regions);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *test_img_path      = NULL;
    char *ref_img_path       = NULL;
    char *ref_label_img_path = NULL;
    char *localizer_path     = NULL;
    char *out_json_path      = NULL;


    iftGetRequiredArgs(args, &test_img_path, &ref_img_path, &ref_label_img_path, &localizer_path, &out_json_path);

    timer *t1 = iftTic();

    puts("- Reading MRI Test Image");
    iftImage *test_img = iftReadImageByExt(test_img_path);

    puts("- Reading MRI Reference Image");
    iftImage *ref_img = iftReadImageByExt(ref_img_path);

    puts("- Reading MRI Reference Label Image");
    iftImage *ref_label_img = iftReadImageByExt(ref_label_img_path);

    puts("- Reading Localizer Mean Center");
    iftBoundingBox *search_regions = NULL;
    iftLabelBoundingBoxArray *lbbs = iftReadLocalizerMeanCenter(localizer_path, &search_regions);

    puts("- Finding the Best Subcortical Bounding Boxes");
    for (int o = 0; o < lbbs->labels->n; o++) {
        printf("###### Object: %d\n", lbbs->labels->val[o]);
        if (!iftDictContainKey("--ignore-local-search", args, NULL)) {
            puts("- Running Local Search");
            lbbs->bbs[o] = iftMSPSMaxSubcorticalNMI(test_img, ref_img, ref_label_img, lbbs->labels->val[o], lbbs->bbs[o], search_regions[o]);
        }
    }

    puts("- Writing the Labeled Subcortical Bounding Boxes");
    iftWriteLabelBoundingBoxArray(lbbs, out_json_path);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(test_img_path);
    iftFree(ref_img_path);
    iftFree(ref_label_img_path);
    iftFree(localizer_path);
    iftFree(out_json_path);

    iftDestroyImage(&test_img);
    iftDestroyImage(&ref_img);
    iftDestroyImage(&ref_label_img);
    iftFree(search_regions);
    iftDestroyLabelBoundingBoxArray(&lbbs);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Find Brain Substructures from bounding boxes with local search";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-t", .long_name = "--test-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="MRI Test Image."},
        {.short_name = "-r", .long_name = "--reference-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="MRI Reference Image."},
        {.short_name = "-l", .long_name = "--reference-label-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="MRI Label Reference Image with the Target Objects."},
        {.short_name = "-z", .long_name = "--localizer-mean-center", .has_arg=true, .arg_type=IFT_STR_TYPE,
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



void iftGetRequiredArgs(  iftDict *args, char **test_img_path, char **ref_img_path, char **ref_label_img_path,
                        char **localizer_path, char **out_json_path) {
    *test_img_path      = iftGetStrValFromDict("--test-image", args);
    *ref_img_path       = iftGetStrValFromDict("--reference-image", args);
    *ref_label_img_path = iftGetStrValFromDict("--reference-label-image", args);
    *localizer_path     = iftGetStrValFromDict("--localizer-mean-center", args);
    *out_json_path      = iftGetStrValFromDict("--output-bb-json", args);


    char *parent_dir = iftParentDir(*out_json_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- MRI Test Image: %s\n", *test_img_path);
    printf("- MRI Reference Image: %s\n", *ref_img_path);
    printf("- MRI Reference Label Image: %s\n", *ref_label_img_path);
    printf("- Localizer Mean Center: %s\n", *localizer_path);
    printf("- Output Json Path: %s\n", *out_json_path);
    puts("-----------------------\n");
}



iftLabelBoundingBoxArray *iftReadLocalizerMeanCenter(const char *json_path, iftBoundingBox **out_search_regions) {
    iftDict *json = iftReadJson(json_path);
    json->erase_elements = true;

    iftIntArray *labels            = iftGetIntArrayFromDict("labels", json);
    iftLabelBoundingBoxArray *lbbs = iftCreateLabelBoundingBoxArray(labels->n);
    iftBoundingBox *search_regions = iftAlloc(labels->n, sizeof(iftBoundingBox));
    iftCopyIntArray(lbbs->labels->val, labels->val, labels->n);

    for (int o = 0; o < labels->n; o++) {
        char key[512];

        sprintf(key, "data:%d:bounding-box:begin", labels->val[o]);
        iftIntArray *begin = iftGetIntArrayFromDict(key, json);
        lbbs->bbs[o].begin.x = begin->val[0];
        lbbs->bbs[o].begin.y = begin->val[1];
        lbbs->bbs[o].begin.z = begin->val[2];

        sprintf(key, "data:%d:bounding-box:end", labels->val[o]);
        iftIntArray *end = iftGetIntArrayFromDict(key, json);
        lbbs->bbs[o].end.x = end->val[0];
        lbbs->bbs[o].end.y = end->val[1];
        lbbs->bbs[o].end.z = end->val[2];

        sprintf(key, "data:%d:search-region-size", labels->val[o]);
        iftIntArray *search_region_sizes = iftGetIntArrayFromDict(key, json);
        iftVoxel bb_center               = iftBoundingBoxCenterVoxel(lbbs->bbs[o]);
        search_regions[o].begin.x = bb_center.x - search_region_sizes->val[0];
        search_regions[o].begin.y = bb_center.y - search_region_sizes->val[1];
        search_regions[o].begin.z = bb_center.z - search_region_sizes->val[2];
        search_regions[o].end.x = bb_center.x + search_region_sizes->val[0];
        search_regions[o].end.y = bb_center.y + search_region_sizes->val[1];
        search_regions[o].end.z = bb_center.z + search_region_sizes->val[2];
    }
    iftDestroyDict(&json);

    *out_search_regions = search_regions;

    return lbbs;
}
/*************************************************************/












