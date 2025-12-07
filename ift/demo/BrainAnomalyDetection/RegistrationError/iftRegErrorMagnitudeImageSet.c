#include "ift.h"
#include "ift/medical/brain/AsymmetryAnalysis.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetInputs(  iftDict *args, char **img_set_entry, char **template_path,
                 char **out_dir, char **mask_path, char **bias_path);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    char *img_set_entry = NULL;
    char *template_path = NULL;
    char *out_dir = NULL;
    // optional aegs
    char *mask_path = NULL;
    char *bias_path = NULL;

    iftGetInputs(args, &img_set_entry, &template_path, &out_dir, &mask_path, &bias_path);

    timer *t1 = iftTic();
    
    iftFileSet *img_set = iftLoadFileSetFromDirOrCSV(img_set_entry, 0, true);
    iftImage *template_img = iftReadImageByExt(template_path);

    iftImage *mask = (mask_path) ? iftReadImageByExt(mask_path) : NULL;
    iftImage *bias = (bias_path) ? iftReadImageByExt(bias_path) : NULL;
    
    #pragma omp parallel for
    for (int f = 0; f < img_set->n; f++) {
        const char *img_path = img_set->files[f]->path;
        char *filename = iftFilename(img_path, NULL);
        char *out_reg_error_mag_path = iftJoinPathnames(2, out_dir, filename);
        printf("[%d/%ld]\nimage: %s\nout reg error: %s\n\n", f, img_set->n - 1, img_path, out_reg_error_mag_path);

        iftImage *img = iftReadImageByExt(img_path);
        iftImage *reg_error_mag = iftRegErrorMagnitude(img, template_img, bias);
        if (mask) {
            iftImage *aux = reg_error_mag;
            reg_error_mag = iftMask(reg_error_mag, mask);
            iftDestroyImage(&aux);
        }

        iftWriteImageByExt(reg_error_mag, out_reg_error_mag_path);

        iftFree(filename);
        iftFree(out_reg_error_mag_path);
        iftDestroyImage(&img);
        iftDestroyImage(&reg_error_mag);
    }

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyFileSet(&img_set);
    iftDestroyImage(&template_img);
    iftDestroyImage(&mask);
    iftDestroyImage(&bias);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] =
        "- For an image set, computes the Magnitude of the Registration Error --- voxel-wise absolute difference -- between "
        "each registered image and its template and attenuate them based on the EDT for objects in a segmentation mask.\n"
        "- The output registration error image is saved into the output directory with the same image filename.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--image-set-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Directory or CSV file with the pathnames from registered images."},
        {.short_name = "-t", .long_name = "--template", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Template (Reference Image)."},
        {.short_name = "-o", .long_name = "--out-dir", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output directory where the registration errors are saved."},
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


void iftGetInputs(  iftDict *args, char **img_set_entry, char **template_path,
                 char **out_dir, char **mask_path, char **bias_path) {
    *img_set_entry = iftGetStrValFromDict("--image-set-entry", args);
    *template_path = iftGetStrValFromDict("--template", args);
    *out_dir = iftGetStrValFromDict("--out-dir", args);

    if (iftDictContainKey("--mask", args, NULL))
        *mask_path = iftGetStrValFromDict("--mask", args);

    if (iftDictContainKey("--bias", args, NULL))
        *bias_path = iftGetStrValFromDict("--bias", args);
    else *bias_path = NULL;

    puts("--------------------");
    printf("- Image Set Entry: %s\n", *img_set_entry);
    printf("- Template Path: %s\n", *template_path);
    printf("- Output Dir: %s\n", *out_dir);
    puts("--------------------");
    if (*mask_path)
        printf("- Mask: %s\n", *mask_path);
    if (*bias_path)
        printf("- Bias: %s\n", *bias_path);
    puts("--------------------\n");
}
/*************************************************************/
