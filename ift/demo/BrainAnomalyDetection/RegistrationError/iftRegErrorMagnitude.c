#include "ift.h"
#include "ift/medical/brain/AsymmetryAnalysis.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetInputs(  iftDict *args, char **img_path, char **template_path, char **out_path, char **mask_path,
                  char **bias_path);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    char *img_path = NULL;
    char *template_path = NULL;
    char *out_path = NULL;
    char *mask_path = NULL;
    char *bias_path = NULL;


    iftGetInputs(args, &img_path, &template_path, &out_path, &mask_path, &bias_path);
    
    timer *t1 = iftTic();
    
    iftImage *mask = (mask_path) ? iftReadImageByExt(mask_path) : NULL;
    iftImage *bias = (bias_path) ? iftReadImageByExt(bias_path) : NULL;

    
    iftImage *img = iftReadImageByExt(img_path);
    iftImage *template_img = iftReadImageByExt(template_path);
    iftImage *reg_error_mag = iftRegErrorMagnitude(img, template_img, bias);
    if (mask) {
        iftImage *aux = reg_error_mag;
        reg_error_mag = iftMask(reg_error_mag, mask);
        iftDestroyImage(&aux);
    }
    iftWriteImageByExt(reg_error_mag, out_path);


    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyImage(&img);
    iftDestroyImage(&template_img);
    iftDestroyImage(&reg_error_mag);
    iftDestroyImage(&mask);
    iftDestroyImage(&bias);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Computes the Magnitude of the Registration Error --- voxel-wise absolute difference -- between " \
        "a registered image and its template.\n" \
        "- A Bias image (e.g. the normal reg. error magnitude) can be passed to attenuate the registration error " \
        "Magnitude\n." \
        "- If a mask is passed, the resulting registration error mag. will be only considered inside it.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--image-path", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the registered Image."},
        {.short_name = "-t", .long_name = "--template", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Template (Reference Image)."},
        {.short_name = "-o", .long_name = "--out-reg-error-mag-map", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname for the resulting registration error magnitude."},
        {.short_name = "-m", .long_name = "--mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Pathname with the mask of the target objects."},
        {.short_name = "-b", .long_name = "--bias", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Pathname from the bias (e.g. normal reg error magnitude) used to attenuate the " \
                                "resulting registration error magnitude."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetInputs(  iftDict *args, char **img_path, char **template_path, char **out_path, char **mask_path,
                  char **bias_path) {
    *img_path = iftGetStrValFromDict("--image-path", args);
    *template_path = iftGetStrValFromDict("--template", args);
    *out_path = iftGetStrValFromDict("--out-reg-error-mag-map", args);
    
    if (iftDictContainKey("--mask", args, NULL))
        *mask_path = iftGetStrValFromDict("--mask", args);

    if (iftDictContainKey("--bias", args, NULL))
        *bias_path = iftGetStrValFromDict("--bias", args);
    else *bias_path = NULL;

    puts("--------------------");
    printf("- Image Path: %s\n", *img_path);
    printf("- Template Path: %s\n", *template_path);
    printf("- Output Image Path: %s\n", *out_path);
    puts("--------------------");
    if (*mask_path)
        printf("- Mask: %s\n", *mask_path);
    if (*bias_path)
        printf("- Normal asymmetry Map: %s\n", *bias_path);
    puts("--------------------\n");
}
/*************************************************************/


