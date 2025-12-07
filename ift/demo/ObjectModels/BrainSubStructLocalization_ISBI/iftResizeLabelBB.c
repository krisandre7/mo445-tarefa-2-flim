#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **bb_json_path, char **sizes_json_path, char **out_bb_json_path);

#include "iftReadWriteBBSizes.c" // file with the function iftReadBBSizes and iftWriteBBSizes
/*************************************************************/




int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *bb_json_path     = NULL;
    char *sizes_json_path  = NULL;
    char *out_bb_json_path = NULL;

    iftGetRequiredArgs(args, &bb_json_path, &sizes_json_path, &out_bb_json_path);

    timer *t1 = iftTic();


    puts("- Reading Input Label Bounding Boxes");
    iftLabelBoundingBoxArray *lbbs = iftReadLabelBoundingBoxArray(bb_json_path);
    iftDict *labels_dict_idxs = iftIntArrayToDict(lbbs->labels->val, lbbs->labels->n);


    puts("- Reading BB Size File");
    iftIntArray *labels = NULL;
    iftIntMatrix *size_mat = iftReadBBSizes(sizes_json_path, &labels);

    puts("- Resizing BBs");
    for (int o = 0; o < labels->n; o++) {
        int label_idx = iftGetLongValFromDict(labels->val[o], labels_dict_idxs);

        iftBoundingBox size_bb;
        size_bb.begin.x = size_bb.begin.y = size_bb.begin.z = 0;
        size_bb.end.x   = iftMatrixElem(size_mat, 0, o) - 1;
        size_bb.end.y   = iftMatrixElem(size_mat, 1, o) - 1;
        size_bb.end.z   = iftMatrixElem(size_mat, 2, o) - 1;

        iftVoxel bb_center = iftBoundingBoxCenterVoxel(lbbs->bbs[label_idx]);
        lbbs->bbs[label_idx] = iftCentralizeBoundingBox(size_bb, bb_center);
    }

    puts("- Writing Output Resized Label Bounding Boxes");
    iftWriteLabelBoundingBoxArray(lbbs, out_bb_json_path);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(bb_json_path);
    iftFree(sizes_json_path);
    iftFree(out_bb_json_path);
    iftDestroyLabelBoundingBoxArray(&lbbs);
    iftDestroyIntArray(&labels);
    iftDestroyIntMatrix(&size_mat);
    iftDestroyDict(&labels_dict_idxs);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Resize a Label Bounding Box given a json file with the size of each bounding box.\n" \
        "- The bounding box's centers keeps the same";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--input-bb-json", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Input Json File with the Labeled Bounding Boxes."},
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


void iftGetRequiredArgs(  iftDict *args, char **bb_json_path, char **sizes_json_path, char **out_bb_json_path) {

    *bb_json_path     = iftGetStrValFromDict("--input-bb-json", args);
    *sizes_json_path  = iftGetStrValFromDict("--sizes-json", args);
    *out_bb_json_path = iftGetStrValFromDict("--output-bb-json", args);

    char *parent_dir = iftParentDir(*out_bb_json_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- Bounding Box Json File: \"%s\"\n", *bb_json_path);
    printf("- Json File with the BB sizes: \"%s\"\n", *sizes_json_path);
    printf("- Output Resized Bounding Box Json File: \"%s\"\n", *out_bb_json_path);
    puts("-----------------------\n");
}
/*************************************************************/












