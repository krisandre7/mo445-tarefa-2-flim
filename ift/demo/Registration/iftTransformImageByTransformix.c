#include "ift.h"
#include "ift/medical/registration/Elastix.h"

/************************** HEADERS **************************/
iftDict *iftGetArguments(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, char **def_fields_path, char **out_img_path);
/*************************************************************/



int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArguments(argc, argv);   

    // mandatory args
    char *img_path        = NULL;
    char *def_fields_path = NULL;
    char *out_img_path    = NULL;

    iftGetRequiredArgs(args, &img_path, &def_fields_path, &out_img_path);

    timer *t1 = iftTic();

    puts("- Reading Image");
    iftImage *img = iftReadImageByExt(img_path);

    puts("- Transforming/Mapping Image");
    iftImage *mapped_img = iftTransformImageByTransformix(img, def_fields_path);        

    puts("- Writing Mapped Image");
    iftWriteImageByExt(mapped_img, out_img_path);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));

    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(img_path);
    iftFree(def_fields_path);
    iftFree(out_img_path);
    iftDestroyImage(&img);
    iftDestroyImage(&mapped_img);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArguments(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- This program transforms/maps an Image or a Label Image (with labels in [0, num-objects]) to a given Coordinate Space " \
        "using the program Transformix.\n" \
        "- The program uses the Deformation Fields obtained via Elastix program to transform the Input Image "\
        "to the Target Coordinate Space.\n" \
        "- To obtain the Deformation Field files, run either the program iftRegisterImageByElatix.c or iftRegisterImageSetByElastix.c\n\n" \
        "* PS1: Pass only the last Deformation Field File resulting from the program.\n" \
        "Ex: If the registration outputted 3 deformation files (xxx_DefFields.0.txt, xxx_DefFields.1.txt, xxx_DefFields.2.txt), " \
        "use only the last one (xxx_DefFields.2.txt)\n";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--input-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Input (Label) Image to be transformed/mapped."},
        {.short_name = "-t", .long_name = "--def-fields-file", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Deformation Fields File"},
        {.short_name = "-o", .long_name = "--output-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output image pathname."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);
    
    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **img_path, char **def_fields_path, char **out_img_path) {
    *img_path        = iftGetStrValFromDict("--input-img", args);
    *def_fields_path = iftGetStrValFromDict("--def-fields-file", args);
    *out_img_path    = iftGetStrValFromDict("--output-img", args);

    char *parent_dir = iftParentDir(*out_img_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- Input Image: \"%s\"\n", *img_path);
    printf("- Deformation Field File: \"%s\"\n", *def_fields_path);
    printf("- Output Image: \"%s\"\n", *out_img_path);
    puts("-----------------------");
}

/*************************************************************/










