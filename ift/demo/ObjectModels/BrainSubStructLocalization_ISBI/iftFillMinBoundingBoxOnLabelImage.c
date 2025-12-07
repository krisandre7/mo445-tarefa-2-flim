#include "ift.h"

/************************** HEADERS **************************/
typedef enum {
    X_AXIS, Y_AXIS, Z_AXIS
} iftAxis;

iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **label_path, char **bb_json_path, char **out_path);
iftImage *iftFillMinBoundingBoxLabelImage(  iftImage *label_img,   iftLabelBoundingBoxArray *lbbs);
/*************************************************************/



int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *label_path   = NULL;
    char *bb_json_path = NULL;
    char *out_path     = NULL;

    iftGetRequiredArgs(args, &label_path, &bb_json_path, &out_path);

    timer *t1 = iftTic();

    puts("- Reading Input Binary/Labeled Image");
    iftImage *label_img = iftReadImageByExt(label_path);

    puts("- Reading Label BBs");
    iftLabelBoundingBoxArray *lbbs = iftReadLabelBoundingBoxArray(bb_json_path);

    puts("- Getting Min Bounding Box Image");
    iftImage *out_mbb_img = iftFillMinBoundingBoxLabelImage(label_img, lbbs);

    puts("- Writing Output Min Bounding Box Image");
    iftWriteImageByExt(out_mbb_img, out_path);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));


    iftDestroyDict(&args);
    iftFree(label_path);
    iftFree(out_path);
    iftDestroyImage(&label_img);
    iftDestroyImage(&out_mbb_img);
    iftDestroyLabelBoundingBoxArray(&lbbs);

    return 0;
}



/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Fill the Min Bounding Boxes in Label Images.";


    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-l", .long_name = "--label-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Input Labeled Image."},
        {.short_name = "-i", .long_name = "--input-bb-json", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Input Json File with the Labeled Bounding Boxes."},
        {.short_name = "-o", .long_name = "--output-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Min Bounding Box Image."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **label_path, char **bb_json_path, char **out_path) {
    *label_path   = iftGetStrValFromDict("--label-img", args);
    *bb_json_path = iftGetStrValFromDict("--input-bb-json", args);
    *out_path     = iftGetStrValFromDict("--output-img", args);

    if (!iftFileExists(*label_path))
        iftError("Input Binary/Labeled Image %s does not exist", "iftGetRequiredArgs", *label_path);

    char *parent_dir = iftParentDir(*out_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);


    puts("-----------------------");
    printf("- Input Binary/Labeled Image: %s\n", *label_path);
    printf("- Bounding Box Json File: \"%s\"\n", *bb_json_path);
    printf("- Output Minimum Bounding Box Image: %s\n", *out_path);
    puts("-----------------------\n");
}


iftImage *iftFillMinBoundingBoxLabelImage(  iftImage *label_img,   iftLabelBoundingBoxArray *lbbs) {
    iftImage *mbb_img = iftCreateImageFromImage(label_img);

    for (int o = 0; o < lbbs->labels->n; o++)
        iftFillBoundingBoxInImage(mbb_img, lbbs->bbs[o], lbbs->labels->val[o]);

    iftImage *add_img = iftAdd(label_img, mbb_img);
    
    iftDestroyImage(&mbb_img);

    return add_img;
}







