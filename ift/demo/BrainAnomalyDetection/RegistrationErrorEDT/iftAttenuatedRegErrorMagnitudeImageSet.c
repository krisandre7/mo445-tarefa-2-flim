#include "ift.h"
#include "ift/medical/brain/AsymmetryAnalysis.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_set_entry, char **template_path, char **label_img_path,
                        char **out_dir);
void iftGetOptionalArgs(  iftDict *args, float *max_attenuation_factor, char **function, float *exponent);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    char *img_set_entry = NULL;
    char *template_path = NULL;
    char *label_img_path = NULL;
    char *out_dir = NULL;
    // optional aegs
    float max_attenuation_factor = 0.5;
    char *function = NULL;
    float exponent = 2.0;

    iftGetRequiredArgs(args, &img_set_entry, &template_path, &label_img_path, &out_dir);
    iftGetOptionalArgs(args, &max_attenuation_factor, &function, &exponent);

    timer *t1 = iftTic();
    
    iftFileSet *img_set = iftLoadFileSetFromDirOrCSV(img_set_entry, 0, true);
    iftImage *template_img = iftReadImageByExt(template_path);
    iftImage *label_img = iftReadImageByExt(label_img_path);
    
    puts("- Computing weights based on EDT");
    iftFImage *weights = NULL;
    if (iftCompareStrings(function, "none")) {
        iftFImage *aux = iftCreateConstantFImage(label_img->xsize, label_img->ysize, label_img->zsize, 1.0);
        weights = iftFMask(aux, label_img);
        iftDestroyFImage(&aux);
    }
    else if (iftCompareStrings(function, "linear")) {
        weights = iftComputeLinearAttenuationWeightsByEDT(label_img, max_attenuation_factor);
    }
    else if (iftCompareStrings(function, "exp")) {
        weights = iftComputeExponentialAttenuationWeightsByEDT(label_img, max_attenuation_factor, exponent);
    }
    else { iftError("Invalid Attenuation Function: %s\nTry none, linear or exp", "main", function); }

    
    #pragma omp parallel for
    for (int f = 0; f < img_set->n; f++) {
        const char *img_path = img_set->files[f]->path;
        char *filename = iftFilename(img_path, NULL);
        char *out_reg_error_path = iftJoinPathnames(2, out_dir, filename);
        printf("[%d/%ld]\nimage: %s\nout reg error: %s\n\n", f, img_set->n - 1, img_path, out_reg_error_path);

        iftImage *img = iftReadImageByExt(img_path);
        iftImage *reg_error = iftWeightedRegErrorMagnitude(img, template_img, weights);
        iftWriteImageByExt(reg_error, out_reg_error_path);

        iftFree(filename);
        iftFree(out_reg_error_path);
        iftDestroyImage(&img);
        iftDestroyImage(&reg_error);
    }

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyFileSet(&img_set);
    iftDestroyImage(&template_img);
    iftDestroyImage(&label_img);
    iftDestroyFImage(&weights);

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
        {.short_name = "-m", .long_name = "--label-image-path", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname with the Label Image of the target objects."},
        {.short_name = "-o", .long_name = "--out-dir", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output directory where the registration errors are saved."},
        {.short_name = "-f", .long_name = "--attenuation-function", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Attenuation Function: [none, linear, exp]. Default: linear"},
        {.short_name = "-a", .long_name = "--max-attenuate-factor", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Maximum Attenuate Factor used to attenuate the errors based on the EDT. Default: 0.75. " \
                                "Not used for 'none' function"}, 
        {.short_name = "-e", .long_name = "--exponent", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Exponent for the exponent attenuation function. Default: 2.0. Not used for linear function."},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **img_set_entry, char **template_path, char **label_img_path,
                        char **out_dir) {
    *img_set_entry = iftGetStrValFromDict("--image-set-entry", args);
    *template_path = iftGetStrValFromDict("--template", args);
    *label_img_path = iftGetStrValFromDict("--label-image-path", args);
    *out_dir = iftGetStrValFromDict("--out-dir", args);

    if (!iftDirExists(*out_dir)) { iftMakeDir(*out_dir); }

    puts("--------------------");
    printf("- Image Set Entry: %s\n", *img_set_entry);
    printf("- Template Path: %s\n", *template_path);
    printf("- Label Image: %s\n", *label_img_path);
    printf("- Output Dir: %s\n", *out_dir);
    puts("--------------------");
}


void iftGetOptionalArgs(  iftDict *args, float *max_attenuation_factor, char **function, float *exponent) {
    if (iftDictContainKey("--max-attenuate-factor", args, NULL)) {
        *max_attenuation_factor = iftGetDblValFromDict("--max-attenuate-factor", args);
    }
    else { *max_attenuation_factor = 0.5; }

    if (iftDictContainKey("--attenuation-function", args, NULL)) {
        *function = iftGetStrValFromDict("--attenuation-function", args);

        if (iftDictContainKey("--exponent", args, NULL)) {
            *exponent = iftGetDblValFromDict("--exponent", args);
        }
        else { *exponent = 2.0; }
    }
    else { *function = iftCopyString("linear"); }

    printf("- Attenuation Function: %s\n", *function);

    if (!iftCompareStrings(*function, "none")) {
        printf("- Max Attenuation Factor: %f\n", *max_attenuation_factor);
    }
    if (iftCompareStrings(*function, "exp")) {
        printf("- Exponent: %f\n", *exponent);
    }
    puts("--------------------\n");
}
/*************************************************************/
