#include "ift.h"

/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **label_path, char **out_path);
/*************************************************************/

int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    char *label_path = NULL;
    char *out_path = NULL;

    iftGetRequiredArgs(args, &label_path, &out_path);

    timer *t1 = iftTic();

    puts("- Reading Input Binary/Labeled Image");
    iftImage *label_img = iftReadImageByExt(label_path);

    puts("- Relabeled Image");
    int n_objs = 0;
    iftImage *out_img = iftConnCompLabeling(label_img, &n_objs);
    printf("- Number of objects after relabeling: %d\n", n_objs);

    puts("- Writing Output Image");
    iftWriteImageByExt(out_img, out_path);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));


    iftDestroyDict(&args);
    iftFree(label_path);
    iftFree(out_path);
    iftDestroyImage(&label_img);
    iftDestroyImage(&out_img);

    return 0;
}



/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Label the connected components (objects) of a binary or labeled image.";


    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--label-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Binary or Labeled Image to be relabeled."},
        {.short_name = "-o", .long_name = "--output-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Relabeled Image."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **label_path, char **out_path) {
    *label_path = iftGetStrValFromDict("--label-img", args);
    *out_path   = iftGetStrValFromDict("--output-img", args);

    if (!iftFileExists(*label_path))
        iftError("Input Binary/Labeled Image %s does not exist", "iftGetRequiredArgs", *label_path);

    char *parent_dir = iftParentDir(*out_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);


    puts("-----------------------");
    printf("- Input Binary/Labeled Image: %s\n", *label_path);
    printf("- Output Relabeled Image: %s\n", *out_path);
    puts("-----------------------\n");
}








