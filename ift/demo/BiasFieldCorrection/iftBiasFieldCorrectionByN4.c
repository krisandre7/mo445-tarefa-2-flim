/**
 * @file
 * @brief Correct the Inhomogeneity of field in a MRI Image using N4 algorithm.
 * @note See the source code in @ref iftCorrectInhomogeneiryByN4.c
 *
 * @example iftCorrectInhomogeneiryByN4.c
 * @brief Correct the Inhomogeneity of field in a MRI Image using N4 algorithm.
 * @author Samuel Martins
 * @date Jan 16, 2017
 */



#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, char **out_path);
void iftGetOptionalArgs(  iftDict *args, char **mask_path, int *shrink_factor, bool *save_bias_field_image);
/*************************************************************/


int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);
    
    // mandatory args
    char *img_path = NULL;
    char *out_path = NULL;

    // optional args
    char *mask_path = NULL;
    int shrink_factor;
    bool save_bias_field_image = false;

    iftGetRequiredArgs(args, &img_path, &out_path);
    iftGetOptionalArgs(args, &mask_path, &shrink_factor, &save_bias_field_image);

    timer *t1 = iftTic();
    
    puts("- Reading Input Image");
    iftImage *img = iftReadImageByExt(img_path);

    iftImage *mask = NULL;
    if (mask_path != NULL) {
        puts("- Reading Binary Mask Image");
        mask = iftReadImageByExt(mask_path);
    }

    printf("- Applying N4 Inhomogeneity Correction\n");
    iftImage *out_bias = NULL;

    timer *tn4 = iftTic();

    iftImage *out = iftN4BiasFieldCorrection(img, mask, shrink_factor, &out_bias);

    printf("\n-> Time N4: %s\n\n", iftFormattedTime(iftCompTime(tn4, iftToc())));    

    if (save_bias_field_image) {
        printf("- Writing Output Bias Field Image\n");
        char *out_bias_path = iftConcatStrings(3, iftBasename(out_path), "_bias", iftFileExt(out_path));
        iftWriteImageByExt(out_bias, out_bias_path);
        iftFree(out_bias_path);
    }
    
    printf("- Writing Output Image\n");
    iftWriteImageByExt(out, out_path);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));

    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(img_path);
    iftFree(out_path);
    iftFree(mask_path);
    iftDestroyImage(&img);
    iftDestroyImage(&out);
    iftDestroyImage(&out_bias);

    return 0;
}



/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Apply the N4 algorithm [1] for Inhomogeneity Correct in a MRI Image.\n" \
        "- It is used default parameters.\n\n" \
        "[1] Tustison, Nicholas J., et al. \"N4ITK: improved N3 bias correction.\" IEEE transactions on medical imaging 29.6 (2010): 1310-1320.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--input-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Input Image to be corrected."},
        {.short_name = "-o", .long_name = "--output-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the output image corrected by N4."},
        {.short_name = "-m", .long_name = "--mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Binary mask that defines the structure of your interest."},
        {.short_name = "-s", .long_name = "--shrink-factor", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=false, .help="Image Resampling/Shrinking Factor (1, 2, 3, 4, ...) to decrease the computation time.\n" \
                                "Shrink factors <= 4 are commonly used.\n" \
                                "Default: 4"},
        {.short_name = "-b", .long_name = "--save-bias-field-image",
         .required=false, .help="Save the Bias Fields applied by N4. Its pathname will have the same " \
                                "output image's basename with the suffix \"_bias\""}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **img_path, char **out_path) {
    *img_path = iftGetStrValFromDict("--input-img", args);
    *out_path = iftGetStrValFromDict("--output-img", args);

    if (!iftFileExists(*img_path))
        iftError("Input image %s does not exists", "iftGetRequiredArgs", *img_path);

    puts("-----------------------");
    printf("- Input Image: %s\n", *img_path);
    printf("- Output Image: %s\n", *out_path);
    puts("-----------------------");
}


void iftGetOptionalArgs(  iftDict *args, char **mask_path, int *shrink_factor, bool *save_bias_field_image) {
    if (iftDictContainKey("--mask", args, NULL)) {
        *mask_path = iftGetStrValFromDict("--mask", args);
        if (!iftFileExists(*mask_path))
            iftError("Binary Mask %s does not exist", "iftGetOptionalArgs", *mask_path);
    }
    else *mask_path = NULL;

    if (iftDictContainKey("--shrink-factor", args, NULL))
        *shrink_factor = iftGetLongValFromDict("--shrink-factor", args);
    else *shrink_factor = 4;

    *save_bias_field_image = iftDictContainKey("--save-bias-field-image", args, NULL);


    if (*mask_path != NULL) printf("- Binary Mask: %s\n", *mask_path);
    printf("- Shring Factor: %d\n", *shrink_factor);
    printf("- Save Bias Field Image?: %s\n", iftBoolAsString(*save_bias_field_image));
    puts("-----------------------\n");
}





