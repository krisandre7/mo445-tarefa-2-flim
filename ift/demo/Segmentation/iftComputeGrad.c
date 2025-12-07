/**
 * @file
 * @brief Computes the gradient for a Image as a linear combination of the gradient from two Input Images.
 * The resulting gradient might be: \n
 * (1) only the gradient from the first input image(function iftImageBasins())\n
 * (2) a linear combination of the gradient from the first input image with the gradient of 
 * a second input image (eg. an object map), according to the formula: \n
 * grad = (alpha * input_grad_img1) + ((1-alpha) * input_grad_img2)\n\n
 * ps: The option (2) only happens if a second input image is passed for the program.
 * @note See the source code in @ref iftComputeGrad.c
 *
 * @example iftComputeGrad.c
 * @brief Computes the gradient for a Image as a linear combination of the gradient from two Input Images.
 * The resulting gradient might be: \n
 * (1) only the gradient from the first input image(function iftImageBasins())\n
 * (2) a linear combination of the gradient from the first input image with the gradient of 
 * a second input image (eg. an object map), according to the formula: \n
 * grad = (alpha * input_grad_img1) + ((1-alpha) * input_grad_img2)\n\n
 * ps: The option (2) only happens if a second input image is passed for the program.
 * @note See the source code in @ref iftComputeGrad.c
 * @author Samuel Martins
 * @date Jul 1, 2016
 */


#include "ift.h"

/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img1_path, char **out_grad_path);
void iftValidateRequiredArgs(const char *img1_path, const char *out_grad_path);
void iftGetOptionalArgs(  iftDict *args, char **img2_path, float *alpha);
void iftValidateOptionalArgs(const char *img2_path, float alpha);
void iftValidateInputImages(  iftImage *img1,   iftImage *img2);
int iftGetImageMaxRange(  iftDict *args,   iftImage *img);
/*************************************************************/



int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *img1_path      = NULL;
    char *out_grad_path = NULL;
    // optional args
    char *img2_path  = NULL;
    float alpha;

    iftGetRequiredArgs(args, &img1_path, &out_grad_path);
    iftGetOptionalArgs(args, &img2_path, &alpha);

    timer *t1 = iftTic();

    iftImage *img1           = iftReadImageByExt(img1_path);
    int img_range            = iftGetImageMaxRange(args, img1);
    iftImage *grad_img1      = iftImageBasins(img1, NULL);

    iftImage *out_grad_img = NULL;

    if (img2_path == NULL) {
        puts("- Only the Gradient of the First Input Image");
        int max_grad_img1 = iftMaximumValue(grad_img1);

        if (max_grad_img1 > img_range) {
            printf("* Max. Value of the Gradient is greater than the Maximum Range: %d > %d\n",
                   max_grad_img1, img_range);
            puts("- Applying a Linear Stretch\n");
            out_grad_img = iftLinearStretch(grad_img1, iftMinimumValue(grad_img1), max_grad_img1,
                                            0, img_range);
            iftDestroyImage(&grad_img1);
        }
        else out_grad_img = grad_img1;
    }
    else {
        puts("- Linear Combination of Gradients");
        iftImage *img2 = iftReadImageByExt(img2_path);
        iftValidateInputImages(grad_img1, img2);
        
        iftImage *norm_grad_img1 = iftLinearStretch(grad_img1, iftMinimumValue(grad_img1), iftMaximumValue(grad_img1),
                                                    0, img_range);
        iftDestroyImage(&grad_img1);

        iftImage *grad_img2      = iftImageBasins(img2, NULL);
        iftImage *norm_grad_img2 = iftLinearStretch(grad_img2, iftMinimumValue(grad_img2), iftMaximumValue(grad_img2),
                                                    0, img_range);
        iftDestroyImage(&grad_img2);

        out_grad_img = iftImageLinearCombination(norm_grad_img1, norm_grad_img2, alpha);

        iftDestroyImage(&img2);
        iftDestroyImage(&norm_grad_img2);
        iftDestroyImage(&norm_grad_img1);
        iftFree(img2_path);
    }
    // printf("- out: min = %d, max = %d\n", iftMinimumValue(out_grad_img), iftMaximumValue(out_grad_img));


    puts("- Writing Gradient Image");
    iftWriteImageByExt(out_grad_img, out_grad_path);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));


    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyImage(&out_grad_img);
    iftFree(img1_path);
    iftFree(out_grad_path);

    return 0;
}



/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "Computes the gradient for a Image as a linear combination of the gradient from two Input Images.\n" \
        "The resulting gradient might be:\n" \
        "(1) only the gradient from the first input image(function iftImageBasins())\n" \
        "(2) a linear combination of the gradient from the first input image with the gradient of " \
        "a second input image (eg. an object map), according to the formula: \n" \
        "grad = (alpha * input_grad_img1) + ((1-alpha) * input_grad_img2)\n\n" \
        "ps: The option (2) only happens if a second input image is passed for the program.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--first-input-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="First Input Image used to compute the Gradient."},
        {.short_name = "-o", .long_name = "--output-grad-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the Output Gradient Image."},
        {.short_name = "-j", .long_name = "--second-input-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Second Input Image whose gradient is used for the linear combination of gradients."},
        {.short_name = "-a", .long_name = "--alpha", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Factor used to combine linearly the Gradients [0, 1]\n" \
                                "Only used if the second input image is passed.\n" \
                                "Default: 0.5"},
        {.short_name = "-b", .long_name = "--img-depth", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=false, .help="Depth from the Input Image and Object Map in bits (8, 12, 16, ...)\n" \
                                "Default: it tries to find the image depth from the First Input Image automatically"}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **img1_path, char **out_grad_path) {
    *img1_path      = iftGetStrValFromDict("--first-input-img", args);
    *out_grad_path = iftGetStrValFromDict("--output-grad-img", args);

    // iftValidateRequiredArgs(*img1_path, *out_grad_path);

    puts("-----------------------");
    printf("- First Input Image: \"%s\"\n", *img1_path);
    printf("- Output Gradient Image: \"%s\"\n", *out_grad_path);
    puts("-----------------------");

}


void iftValidateRequiredArgs(const char *img1_path, const char *out_grad_path) {
    if (!iftIsImageFile(img1_path))
        iftError("Invalid Input Image: \"%s\"", "iftValidateRequiredArgs", img1_path);

    if (!iftIsImagePathnameValid(out_grad_path)) {
        iftError("Invalid Output Grad Image's Pathname: \"%s\"", "iftValidateRequiredArgs", out_grad_path);
    }

    char *parent_dir = iftParentDir(out_grad_path);
    if (!iftDirExists(parent_dir)) {
        iftMakeDir(parent_dir);
    }
    iftFree(parent_dir);
}


void iftGetOptionalArgs(  iftDict *args, char **img2_path, float *alpha) {
    if (iftDictContainKey("--second-input-img", args, NULL)) {
        *img2_path = iftGetStrValFromDict("--second-input-img", args);
        
        if (iftDictContainKey("--alpha", args, NULL)) {
            *alpha = iftGetDblValFromDict("--alpha", args);
        }
        else *alpha = 0.5;

        iftValidateOptionalArgs(*img2_path, *alpha);

        printf("- Second Input Image: \"%s\"\n", *img2_path);
        printf("- Linear Factor (alpha): %f\n", *alpha);
        puts("-----------------------\n");
    }
    else {
        *img2_path = NULL;
        puts("");
    }
}


void iftValidateOptionalArgs(const char *img2_path, float alpha) {
    if (!iftIsImageFile(img2_path))
        iftError("Second Input Image: \"%s\"", "iftValidateOptionalArgs", img2_path);

    if ((alpha < 0.0) || (alpha > 1.0))
        iftError("Invalid alpha: %f... Try [0,1]\n", "iftValidateOptionalArgs", alpha);
}


void iftValidateInputImages(  iftImage *img1,   iftImage *img2) {
    iftVerifyImageDomains(img1, img2, "iftValidateInputImages");
    if (!iftIsVoxelSizeEqual(img1, img2))
        iftError("Input Images have different Voxel Sizes\n" \
                 "Image 1 (%lf, %lf, %lf)\n" \
                 "Image 2 (%lf, %lf, %lf)", "iftValidateInputImages", 
                 img1->dx, img1->dy, img1->dz, img2->dx, img2->dy, img2->dz);
}


int iftGetImageMaxRange(  iftDict *args,   iftImage *img) {
    int img_range;

    if (iftDictContainKey("--img-depth", args, NULL)) {
        int img_depth = iftGetLongValFromDict("--img-depth", args);
        if (img_depth <= 0)
            iftError("Invalid Image Depth: %d... Try > 0", "iftGetOptionalArgs", img_depth);

        img_range = (1 << img_depth) - 1; // (2^img_depth) - 1
    }
    else img_range = iftNormalizationValue(iftMaximumValue(img));

    printf("  - Image Range: [0, %d]\n", img_range);

    return img_range;
}



