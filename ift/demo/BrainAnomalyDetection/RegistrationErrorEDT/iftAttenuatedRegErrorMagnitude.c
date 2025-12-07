#include "ift.h"
#include "ift/medical/brain/AsymmetryAnalysis.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, char **template_path, char **label_img_path,
                  char **out_path);
void iftGetOptionalArgs(  iftDict *args, float *max_attenuation_factor, char **function, float *exponent, char **bias_path);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    char *img_path = NULL;
    char *template_path = NULL;
    char *label_img_path = NULL;
    char *out_path = NULL;
    // optional aegs
    float max_attenuation_factor = 0.5;
    char *function = NULL;
    float exponent = 2.0;
    char *bias_path = NULL;

    iftGetRequiredArgs(args, &img_path, &template_path, &label_img_path, &out_path);
    iftGetOptionalArgs(args, &max_attenuation_factor, &function, &exponent, &bias_path);

    timer *t1 = iftTic();
    
    iftImage *img = iftReadImageByExt(img_path);
    iftImage *template_img = iftReadImageByExt(template_path);
    iftImage *label_img = iftReadImageByExt(label_img_path);
    iftImage *bias = (bias_path) ? iftReadImageByExt(bias_path) : NULL;

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

    puts("- Computing Attenuated Registration Error Magnitude");
    iftImage *reg_error = iftWeightedRegErrorMagnitude(img, template_img, weights);
    // iftWriteImageByExt(iftFImageToImage(weights, 255), "test/edt.nii.gz");

    if (bias) {
        #pragma omp parallel for
        for (int p = 0; p < reg_error->n; p++) {
            reg_error->val[p] = iftMax(0, reg_error->val[p] - bias->val[p]);
        }
    }


    iftWriteImageByExt(reg_error, out_path);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyImage(&img);
    iftDestroyImage(&template_img);
    iftDestroyImage(&label_img);
    iftDestroyFImage(&weights);
    iftDestroyImage(&reg_error);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Computes the Magnitude of the Registration Error --- voxel-wise absolute difference -- between " \
        "a registered image and its template and attenuate them based on the EDT for objects in a segmentation mask.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--image-path", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the registered Image."},
        {.short_name = "-t", .long_name = "--template", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Template (Reference Image)."},
        {.short_name = "-m", .long_name = "--label-image-path", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname with the Label Image of the target objects."},
        {.short_name = "-o", .long_name = "--out-reg-error-mag-map", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname for the resulting registration error magnitude."},
        {.short_name = "-f", .long_name = "--attenuation-function", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Attenuation Function: [none, linear, exp]. Default: linear"},
        {.short_name = "-a", .long_name = "--max-attenuation-factor", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Maximum Attenuate Factor used to attenuate the errors based on the EDT. Default: 0.75. " \
                                "Not used for 'none' function"}, 
        {.short_name = "-e", .long_name = "--exponent", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Exponent for the exponent attenuation function. Default: 2.0. Not used for linear function."},
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


void iftGetRequiredArgs(  iftDict *args, char **img_path, char **template_path, char **label_img_path,
                  char **out_path) {
    *img_path = iftGetStrValFromDict("--image-path", args);
    *template_path = iftGetStrValFromDict("--template", args);
    *label_img_path = iftGetStrValFromDict("--label-image-path", args);
    *out_path = iftGetStrValFromDict("--out-reg-error-mag-map", args);

    puts("--------------------");
    printf("- Image Path: %s\n", *img_path);
    printf("- Template Path: %s\n", *template_path);
    printf("- Label Image: %s\n", *label_img_path);
    printf("- Output Path: %s\n", *out_path);
    puts("--------------------");
}

void iftGetOptionalArgs(  iftDict *args, float *max_attenuation_factor, char **function, float *exponent, char **bias_path) {
    if (iftDictContainKey("--max-attenuation-factor", args, NULL)) {
        *max_attenuation_factor = iftGetDblValFromDict("--max-attenuation-factor", args);
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

    if (iftDictContainKey("--bias", args, NULL)) {
        *bias_path = iftGetStrValFromDict("--bias", args);
        printf("- Bias: %s\n", *bias_path);
    }
    else { *bias_path = NULL; }

    puts("--------------------\n");
}
/*************************************************************/
