#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **prob_atlas_path, char **sizes_json_path, char **out_bb_json_path);

#include "iftReadBBSizes.c" // file with the function iftReadBBSizes
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *prob_atlas_path  = NULL;
    char *sizes_json_path  = NULL;
    char *out_bb_json_path = NULL;

    iftGetRequiredArgs(args, &prob_atlas_path, &sizes_json_path, &out_bb_json_path);

    timer *t1 = iftTic();

    puts("- Read Prob. Atlas");
    iftMObjModel *prob_atlas = iftReadSOSM(prob_atlas_path);
    iftDict *labels_dict_idxs = iftIntArrayToDict(prob_atlas->labels->val, prob_atlas->labels->n);

    puts("- Reading BB Size File");
    iftIntArray *labels = NULL;
    iftIntMatrix *size_mat = iftReadBBSizes(sizes_json_path, &labels);

    puts("- Finding BBs");
    iftLabelBoundingBoxArray *lbbs = iftCreateLabelBoundingBoxArray(labels->n);
    iftCopyIntArray(lbbs->labels->val, labels->val, lbbs->labels->n);

    for (int o = 0; o < labels->n; o++) {
        printf("\nObject: %d\n", labels->val[o]);
        int label_idx = iftGetLongValFromDict(labels->val[o], labels_dict_idxs);

        iftBoundingBox size_bb;
        size_bb.begin.x = size_bb.begin.y = size_bb.begin.z = 0;
        size_bb.end.x   = iftMatrixElem(size_mat, 0, o) - 1;
        size_bb.end.y   = iftMatrixElem(size_mat, 1, o) - 1;
        size_bb.end.z   = iftMatrixElem(size_mat, 2, o) - 1;

        iftVoxel ref_voxel = prob_atlas->finder->func(prob_atlas->ref_img, prob_atlas->finder->params);
        iftFImage *model = iftModelOnTestImageDomain(prob_atlas->models[label_idx]->prob_map,
                            prob_atlas->ref_img, ref_voxel, prob_atlas->models[label_idx]->disp_vec);
        iftBoundingBox model_bb = iftFMinBoundingBox(model, NULL);
        iftVoxel model_bb_center = iftBoundingBoxCenterVoxel(model_bb);
        puts("model_bb");
        iftPrintBoundingBox(model_bb);
        printf("model_bb_center: (%d, %d, %d)\n", model_bb_center.x, model_bb_center.y, model_bb_center.z);

        lbbs->bbs[o] = iftCentralizeBoundingBox(size_bb, model_bb_center);
        puts("resulting bb");
        iftPrintBoundingBox(lbbs->bbs[o]);
    }


    puts("- Writing Output Resized Label Bounding Boxes");
    iftWriteLabelBoundingBoxArray(lbbs, out_bb_json_path);


    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyMObjModel(&prob_atlas);
    iftDestroyIntArray(&labels);
    iftDestroyIntMatrix(&size_mat);
    iftDestroyDict(&labels_dict_idxs);


    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Returns the Label Bounding Boxes by placing bounding boxes (drawn by use) in the center of the prob. atlas of the target objects.";

    iftCmdLineOpt cmd_line_opts[] = {
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


void iftGetRequiredArgs(  iftDict *args, char **prob_atlas_path, char **sizes_json_path, char **out_bb_json_path) {

    *prob_atlas_path  = iftGetStrValFromDict("--prob-atlas", args);
    *sizes_json_path  = iftGetStrValFromDict("--sizes-json", args);
    *out_bb_json_path = iftGetStrValFromDict("--output-bb-json", args);

    char *parent_dir = iftParentDir(*out_bb_json_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- Prob. Atlas of The Target Objects: \"%s\"\n", *prob_atlas_path);
    printf("- Json File with the BB sizes: \"%s\"\n", *sizes_json_path);
    printf("- Output Resized Bounding Box Json File: \"%s\"\n", *out_bb_json_path);
    puts("-----------------------\n");
}
/*************************************************************/


