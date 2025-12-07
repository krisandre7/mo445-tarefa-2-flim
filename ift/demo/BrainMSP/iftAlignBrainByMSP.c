#include "ift.h"
#include "ift/medical/brain/MSP.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetInputArgs(  iftDict *args, char **brain_img_path, char **msp_path, char **out_img_path,
                     iftInterpolationType *interp_type);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);
    
    char *brain_img_path = NULL;
    char *msp_path = NULL;
    char *out_img_path = NULL;
    iftInterpolationType interp_type;
    
    iftGetInputArgs(args, &brain_img_path, &msp_path, &out_img_path, &interp_type);
    
    timer *t1 = iftTic();
    
    puts("- Reading Brain Image");
    iftImage *brain_img = iftReadImageByExt(brain_img_path);
    
    puts("- Reading MSP");
    iftPlane *msp = iftReadPlane(msp_path);
    
    puts("- Align Brain Image by its MSP");
    iftImage *aligned_brain_img = iftRotateImageToMSP(brain_img, msp, interp_type);
    
    puts("- Writing Aligned Brain Image");
    iftWriteImageByExt(aligned_brain_img, out_img_path);
    
    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyImage(&brain_img);
    iftDestroyImage(&aligned_brain_img);
    iftDestroyPlane(&msp);
    
    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Aligns a brain image by its MSP.\n" \
        "- The MSP becomes the central sagittal slice from the output aligned image.\n" \
        "- Choose the interpolation types: nn (nearest neighbor) or linear (trilinear)\n" \
        "- Use nn to align label images (segmentation masks), otherwise use linear.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--brain-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Input Brain image."},
        {.short_name = "-p", .long_name = "--msp", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="MSP's Pathname (*.json)."},
        {.short_name = "-o", .long_name = "--output-aligned-brain-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Aligned Image by its MSP."},
        {.short_name = "-t", .long_name = "--interpolation-type", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Interpolation Type: nn (Nearest Neighbor) or linear (Trilinear). Default: linear."},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetInputArgs(  iftDict *args, char **brain_img_path, char **msp_path, char **out_img_path,
                     iftInterpolationType *interp_type) {
    *brain_img_path = iftGetStrValFromDict("--brain-image", args);
    *msp_path = iftGetStrValFromDict("--msp", args);
    *out_img_path = iftGetStrValFromDict("--output-aligned-brain-image", args);
    
    char interp_type_str[64];
    if (iftDictContainKey("--interpolation-type", args, NULL)) {
        strcpy(interp_type_str, iftGetConstStrValFromDict("--interpolation-type", args));
        
        if (iftCompareStrings(interp_type_str, "nn"))
            *interp_type = IFT_NEAREST_NEIGHBOR_INTERPOLATION;
        else if (iftCompareStrings(interp_type_str, "linear"))
            *interp_type = IFT_LINEAR_INTERPOLATION;
        else iftError("Invalid Interpolation type: %s\nTry nn or linear", "iftGetInputArgs", interp_type_str);
    }
    else {
        strcpy(interp_type_str, "linear");
        *interp_type = IFT_LINEAR_INTERPOLATION;
    }
    
    puts("-----------------------");
    printf("- Brain Image: %s\n", *brain_img_path);
    printf("- MSP path: %s\n", *msp_path);
    printf("- Output Aligned Brain Image: %s\n", *out_img_path);
    puts("-----------------------");
    printf("- Interpolation Type: %s\n", interp_type_str);
    puts("-----------------------\n");
}
/*************************************************************/


