#include "ift.h"
#include "ift/medical/brain/MSP.h"



/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetInputArgs(  iftDict *args, char **mov_img_path, char **fix_img_path, char **out_img_path);
/*************************************************************/



int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);
    
    char *mov_img_path = NULL;
    char *fix_img_path = NULL;
    char *out_img_path = NULL;
    
    iftGetInputArgs(args, &mov_img_path, &fix_img_path, &out_img_path);
    
    timer *t1 = iftTic();
    
    puts("- Reading Moving Image");
    iftImage *mov_img = iftReadImageByExt(mov_img_path);
    
    puts("- Reading Moving Image");
    iftImage *fix_img = iftReadImageByExt(fix_img_path);
    
    puts("- Registering");
    iftImage *out_img = iftBrainAffineRegistration(fix_img,mov_img);
    
    puts("- Writing Aligned Brain Image");
    iftWriteImageByExt(out_img, out_img_path);
    
    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyImage(&mov_img);
    iftDestroyImage(&fix_img);
    iftDestroyImage(&out_img);
    
    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Register the Moving image to the Fixed Image using affine transformations.\n" \
        "- Uses the MSPS method (Ruppert et al.)\n";
     
    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-m", .long_name = "--moving-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Input Moving image."},
        {.short_name = "-f", .long_name = "--fixed-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Input Fixed image."},
        {.short_name = "-o", .long_name = "--output-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output registered moving image."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetInputArgs(  iftDict *args, char **mov_img_path, char **fix_img_path, char **out_img_path) {
    *mov_img_path = iftGetStrValFromDict("--moving-image", args);
    *fix_img_path = iftGetStrValFromDict("--fixed-image", args);
    *out_img_path = iftGetStrValFromDict("--output-image", args);
    
    if (!iftIsImagePathnameValid(*mov_img_path))
        iftError("Input Moving Image path %s is invalid", "iftGetInputArgs", *mov_img_path);
    if (!iftIsImagePathnameValid(*fix_img_path))
        iftError("Input Fixed Image path %s is invalid", "iftGetInputArgs", *fix_img_path);
    if (!iftIsImagePathnameValid(*out_img_path))
        iftError("Output Aligned Brain Image path %s is invalid", "iftGetInputArgs", *out_img_path);
    
    
    puts("-----------------------");
    printf("- Moving Image: %s\n", *mov_img_path);
    printf("- Fixed Image: %s\n", *fix_img_path);
    printf("- Output Image: %s\n", *out_img_path);
    puts("-----------------------\n");
}
/*************************************************************/


