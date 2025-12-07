#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **target_label_img_path, char **out_bb_json_path);

#include "iftReadWriteBBSizes.c" // file with the function iftReadBBSizes
/*************************************************************/


int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *target_label_img_path = NULL;
    char *out_bb_json_path      = NULL;

    iftGetRequiredArgs(args, &target_label_img_path, &out_bb_json_path);

    timer *t1 = iftTic();

    puts("- Reading Target Label Image");
    iftImage *label_img = iftReadImageByExt(target_label_img_path);

    puts("- Finding BBs");
    iftIntArray *labels = iftGetObjectLabels(label_img);
    iftLabelBoundingBoxArray *lbbs = iftCreateLabelBoundingBoxArray(labels->n);
    iftCopyIntArray(lbbs->labels->val, labels->val, lbbs->labels->n);

    lbbs->bbs = iftMinLabelsBoundingBox(label_img, labels, NULL);

    puts("- Writing Output Resized Label Bounding Boxes");
    iftWriteLabelBoundingBoxArray(lbbs, out_bb_json_path);


    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));

    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(target_label_img_path);
    iftFree(out_bb_json_path);
    iftDestroyImage(&label_img);
    iftDestroyIntArray(&labels);

    return 0;
}




/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Finds the Label Bounding Boxes by segmenting the target objects.\n" \
        "- The min bounding boxes drawn will be the resulting patches.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-l", .long_name = "--target-label-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Label Image, provided by some segmentation method, with the target objects."},
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


void iftGetRequiredArgs(  iftDict *args, char **target_label_img_path, char **out_bb_json_path) {
    *target_label_img_path    = iftGetStrValFromDict("--target-label-image", args);
    *out_bb_json_path = iftGetStrValFromDict("--output-bb-json", args);

    char *parent_dir = iftParentDir(*out_bb_json_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- Target Label Image: %s\n", *target_label_img_path);
    printf("- Output Resized Bounding Box Json File: \"%s\"\n", *out_bb_json_path);
    puts("-----------------------\n");
}
/*************************************************************/





