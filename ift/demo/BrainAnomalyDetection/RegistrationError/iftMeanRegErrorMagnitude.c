#include "ift.h"
#include "ift/medical/brain/AsymmetryAnalysis.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetInputs(  iftDict *args, iftFileSet **img_set, char **template_path, char **out_path,
                  char **mask_path, bool *use_stdev);

/*************************************************************/



int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    iftFileSet *img_set = NULL;
    char *template_path = NULL;
    char *out_path = NULL;
    char *mask_path = NULL;
    bool use_stdev = false;

    iftGetInputs(args, &img_set, &template_path, &out_path, &mask_path, &use_stdev);
    
    iftImage *template_img  = iftReadImageByExt(template_path);
    iftImage *mask = (mask_path) ? iftReadImageByExt(mask_path) : NULL;

    timer *t1 = iftTic();

    puts("- Computing Mean Registration Error Magnitude");
    iftImage *mean_reg_error_mag = iftMeanRegErrorMagnitude(img_set, template_img, use_stdev);
    if (mask) {
        puts("****************************************");
        iftImage *aux = mean_reg_error_mag;
        mean_reg_error_mag = iftMask(mean_reg_error_mag, mask);
        iftDestroyImage(&aux);
    }

    puts("- Writing Mean Registration Error");
    iftWriteImageByExt(mean_reg_error_mag, out_path);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));


    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyFileSet(&img_set);
    iftDestroyImage(&mean_reg_error_mag);
    iftDestroyImage(&mask);
    iftDestroyImage(&template_img);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Computes the mean Registration Error Magnitude --- voxel-wise absolute difference -- for a set of " \
        "registered images and the used template.\n" \
        "- Optionally, the standard deviation reg. error magnitude can be added to the resulting mean."
        "- If a mask is passed, the output registration error mag. will be only considered inside it.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--image-set", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Directory of CSV with the registered image pathnames."},
        {.short_name = "-t", .long_name = "--template", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Template where all images are registered."},
        {.short_name = "-o", .long_name = "--out-mean-reg-error-mag-map", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname for the resulting mean registration error magnitude."},
        {.short_name = "-m", .long_name = "--mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Pathname with the mask of the target objects."},
        {.short_name = "", .long_name = "--use-stdev", .has_arg=false,
         .required=false, .help="Add/Use the Standard Deviation Registration Error Magnitudes into the output map."},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetInputs(  iftDict *args, iftFileSet **img_set, char **template_path, char **out_path,
                  char **mask_path, bool *use_stdev) {
    const char *img_entry = iftGetConstStrValFromDict("--image-set", args);
    *img_set = iftLoadFileSetFromDirOrCSV(img_entry, 0, true);
    *template_path = iftGetStrValFromDict("--template", args);
    *out_path = iftGetStrValFromDict("--out-mean-reg-error-mag-map", args);
    
    if (iftDictContainKey("--mask", args, NULL))
        *mask_path = iftGetStrValFromDict("--mask", args);

    *use_stdev = iftDictContainKey("--use-stdev", args, NULL);

    puts("-----------------------");
    printf("- Image Entry: %s\n", img_entry);
    printf("- Template Image: %s\n", *template_path);
    printf("- Output Mean Reg. Error Magnitude: %s\n", *out_path);
    puts("-----------------------");
    if (*mask_path)
        printf("- Mask: %s\n", *mask_path);
    printf("- Add/Use Standard Deviation: %s\n", iftBoolAsString(*use_stdev));
    puts("-----------------------\n");
}
/*************************************************************/


