#include "ift.h"
#include "ift/medical/brain/MSP.h"



/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetInputArgs(  iftDict *args, char **brain_img_path, char **out_img_path, char **out_msp_path);
/*************************************************************/



int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);
    
    char *brain_img_path = NULL;
    char *out_img_path = NULL;
    char *out_msp_path = NULL;
    
    iftGetInputArgs(args, &brain_img_path, &out_img_path, &out_msp_path);
    
    timer *t1 = iftTic();
    
    puts("- Reading Brain Image");
    iftImage *brain_img = iftReadImageByExt(brain_img_path);
    
    puts("- Find and Align Brain Image by its MSP");
    iftPlane *msp;
    iftImage *aligned_brain_img = iftAlignBrainByMSP(brain_img, &msp);
    
    int min = iftMinimumValue(aligned_brain_img);
    if (min < 0)
        iftError("Some problem during MSP Alignment (interpolation): Minimum value %d < 0\n", "main", min);
    
    puts("- Writing Aligned Brain Image");
    iftWriteImageByExt(aligned_brain_img, out_img_path);
    
    if (out_msp_path) {
        puts("- Writing MSP");
        iftWritePlane(msp, out_msp_path);
    }
    
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
        "- Finds the Mid-Sagittal Plane (MSP) from an input image and align this image based on it, by using the algorithm [1].\n" \
        "- The found MSP becomes the central sagittal slice from the output aligned image.\n" \
        "- Optionally, the MSP can also be saved for then being used to align other images associated to the input one, " \
        "such as segmentation masks, etc.\n\n" \
        "[1] Ruppert, Guilherme CS, et al. \"A new symmetry-based method for mid-sagittal plane extraction in neuroimages.\" " \
        "Biomedical Imaging: From Nano to Macro, 2011 IEEE International Symposium on. IEEE, 2011.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--brain-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Input Brain image."},
        {.short_name = "-o", .long_name = "--output-aligned-brain-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Aligned Image by its MSP."},
        {.short_name = "-p", .long_name = "--extracted-msp", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Pathname (*.json) from the extracted MSP."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetInputArgs(  iftDict *args, char **brain_img_path, char **out_img_path, char **out_msp_path) {
    *brain_img_path = iftGetStrValFromDict("--brain-image", args);
    *out_img_path = iftGetStrValFromDict("--output-aligned-brain-image", args);
    
    if (!iftIsImagePathnameValid(*brain_img_path))
        iftError("Input Brain Image path %s is invalid", "iftGetInputArgs", *brain_img_path);
    if (!iftIsImagePathnameValid(*out_img_path))
        iftError("Output Aligned Brain Image path %s is invalid", "iftGetInputArgs", *out_img_path);
    
    if (iftDictContainKey("--extracted-msp", args, NULL)) {
        *out_msp_path = iftGetStrValFromDict("--extracted-msp", args);
        if (!iftEndsWith(*out_msp_path, ".json"))
            iftError("Invalid extension for the output MSP path: %s\nTry *.json", "iftGetInputArgs", *out_msp_path);
    }
    else *out_msp_path = NULL;
    
    puts("-----------------------");
    printf("- Brain Image: %s\n", *brain_img_path);
    printf("- Output Aligned Brain Image: %s\n", *out_img_path);
    puts("-----------------------");
    if (*out_msp_path)
        printf("- Output MSP path: %s\n", *out_msp_path);
    puts("-----------------------\n");
}
/*************************************************************/


