#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **input_json_path, char **out_json_path);
void iftGetOptionalArgs(  iftDict *args, double *scale_factor);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *input_json_path = NULL;
    char *out_json_path  = NULL;
    // optional args
    double scale_factor;

    iftGetRequiredArgs(args, &input_json_path, &out_json_path);
    iftGetOptionalArgs(args, &scale_factor);

    timer *t1 = iftTic();

    puts("- Reading Labeled Bounding Boxes");
    iftIntArray *labels = NULL;
    iftBoundingBox *bbs = iftReadLabelBoundingBox(input_json_path, &labels);

    for (int o = 0; o < labels->n; o++) {
        printf("[%d] Object: %d\n", o, labels->val[o]);
        printf("bbs.begin: (%d, %d, %d)\n", bbs[o].begin.x, bbs[o].begin.y, bbs[o].begin.z);
        printf("bbs.end: (%d, %d, %d)\n", bbs[o].end.x, bbs[o].end.y, bbs[o].end.z);

        bbs[o] = iftScaleBoundingBox(bbs[o], scale_factor);

        printf("bbs.begin: (%d, %d, %d)\n", bbs[o].begin.x, bbs[o].begin.y, bbs[o].begin.z);
        printf("bbs.end: (%d, %d, %d)\n\n", bbs[o].end.x, bbs[o].end.y, bbs[o].end.z);
    }

    puts("- Writing the Json File");
    iftWriteLabelBoundingBox(bbs, labels, out_json_path);


    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(input_json_path);
    iftFree(out_json_path);
    iftFree(bbs);
    iftDestroyIntArray(&labels);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Scale Labeled Bounding Boxes from a Json File.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--input-bb-json", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Input Json Path with the Labeled Bounding Boxes."},
        {.short_name = "-o", .long_name = "--output-bb-json", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Json Path with the Labeled Bounding Boxes."},
        {.short_name = "-a", .long_name = "--scale-factor", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Factor to scale the Bounding Boxes. Default: 1.0"}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **input_json_path, char **out_json_path) {
    *input_json_path = iftGetStrValFromDict("--input-bb-json", args);
    *out_json_path  = iftGetStrValFromDict("--output-bb-json", args);

    char *parent_dir = iftParentDir(*out_json_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- Input BB Json Path: \"%s\"\n", *input_json_path);
    printf("- Output BB Json Path: \"%s\"\n", *out_json_path);
    puts("-----------------------");
}


void iftGetOptionalArgs(  iftDict *args, double *scale_factor) {
    if (iftDictContainKey("--scale-factor", args, NULL))
        *scale_factor = iftGetDblValFromDict("--scale-factor", args);
    else *scale_factor = 1.0;

    printf("- Scaling Factor: %lf\n", *scale_factor);
    puts("-----------------------\n");
}
/*************************************************************/






