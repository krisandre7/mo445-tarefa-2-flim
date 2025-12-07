#include "ift.h"

/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **fixed_grad_path, char **mov_grad_path, char **out_matched_grad_path,
                        char **out_disp_path);
void iftValidateRequiredArgs(const char *fixed_grad_path, const char *mov_grad_path, const char *out_matched_grad_path,
                             const char *out_disp_path);
void iftGetOptionalArgs(  iftDict *args, int *n_scales, int *stride);
void iftWriteInfoFile(const char *fixed_grad_path, const char *mov_grad_path, iftVector best_disp,
                      const char *out_disp_path);
/*************************************************************/


int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *fixed_grad_path       = NULL;
    char *mov_grad_path         = NULL;
    char *out_matched_grad_path = NULL;
    char *out_disp_path         = NULL;
    // optional args
    int n_scales = 3;
    int stride   = 2;

    // iftPrintDict(args);
    iftGetRequiredArgs(args, &fixed_grad_path, &mov_grad_path, &out_matched_grad_path, &out_disp_path);
    iftGetOptionalArgs(args, &n_scales, &stride);

    timer *t1 = iftTic();

    puts("- Reading Fixed Image Gradient");
    iftImage *fixed_grad_img = iftReadImageByExt(fixed_grad_path);

    puts("- Reading Moving Model Gradient");
    iftImage *mov_grad_img = iftReadImageByExt(mov_grad_path);

    iftVector best_disp = iftMSPSMatchGradients(fixed_grad_img, mov_grad_img, n_scales, stride);
    printf("* Displacement of the Best Matching: (dx, dy, dz): (%.0f, %.0f, %.0f)\n\n", best_disp.x, best_disp.y, best_disp.z);

    puts("- Writing Resulting Gradient");
    iftImage *out_matched_grad = iftTranslateImageContent(mov_grad_img, best_disp);
    iftWriteImageByExt(out_matched_grad, out_matched_grad_path);

    puts("- Saving Output Json File");
    iftWriteInfoFile(fixed_grad_path, mov_grad_path, best_disp, out_disp_path);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));

    // destroyers
    iftDestroyDict(&args);
    iftFree(fixed_grad_path);
    iftFree(mov_grad_path);
    iftFree(out_matched_grad_path);
    iftFree(out_disp_path);
    iftDestroyImage(&fixed_grad_img);
    iftDestroyImage(&mov_grad_img);

    return 0;
}



/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "Finds the best Matching between a Fixed Gradient and a Moving Gradient using MSPS algorithm.\n" \
        "The best gradient matching (fitness function) has the maximum average between the intersection of the input gradients.\n" \
        "The Moving Gradient Image is translated over the Fixed one.\n\n" \
        "PS1: Both images must have the same domain and voxel sizes.\n" \
        "PS2: The program also stores the Translated Moving Gradient with the best displacement";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-f", .long_name = "--input-fixed-grad", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Fixed Gradient to be matched from the Test Image."},
        {.short_name = "-m", .long_name = "--input-moving-grad", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Moving Gradient from a Probabilist Atlas."},
        {.short_name = "-g", .long_name = "--output-grad", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Resulting (translated) Moving Gradient Image."},
        {.short_name = "-o", .long_name = "--output-displacement-json", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Json File that stores the best displacements used to match the " \
                               "Moving Gradient on to the Fixed Gradient."},
        {.short_name = "-s", .long_name = "--num-of-scales", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=false, .help="Number of Scales used in optimization.\nDefault: 3"},
        {.short_name = "-t", .long_name = "--stride", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=false, .help="Stride used to compute a regular translation (displacement) of the Moving " \
                                "Image's coordinates for each scale.\nDefault: 2"}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **fixed_grad_path, char **mov_grad_path, char **out_matched_grad_path,
                        char **out_disp_path) {
    *fixed_grad_path       = iftGetStrValFromDict("--input-fixed-grad", args);
    *mov_grad_path         = iftGetStrValFromDict("--input-moving-grad", args);
    *out_matched_grad_path = iftGetStrValFromDict("--output-grad", args);
    *out_disp_path         = iftGetStrValFromDict("--output-displacement-json", args);

    iftValidateRequiredArgs(*fixed_grad_path, *mov_grad_path, *out_matched_grad_path, *out_disp_path);

    puts("-----------------------");
    printf("- Input Fixed Gradient: \"%s\"\n", *fixed_grad_path);
    printf("- Input Moving Gradient: \"%s\"\n", *mov_grad_path);
    printf("- Output Matched Gradient: \"%s\"\n", *out_matched_grad_path);
    printf("- Output Displacement File: \"%s\"\n", *out_disp_path);
    puts("-----------------------");
}


void iftValidateRequiredArgs(const char *fixed_grad_path, const char *moving_grad_path, const char *matched_grad_path,
                             const char *out_disp_path) {
    if (!iftIsImageFile(fixed_grad_path))
        iftError("Invalid Input Fixed Gradient: \"%s\"", "iftValidateRequiredArgs", fixed_grad_path);

    if (!iftIsImageFile(moving_grad_path))
        iftError("Invalid Input Moving Gradient: \"%s\"", "iftValidateRequiredArgs", moving_grad_path);

    if (!iftIsImagePathnameValid(matched_grad_path))
        iftError("Invalid Output Gradient Image: \"%s\"", "iftValidateRequiredArgs", matched_grad_path);

    if (!iftEndsWith(out_disp_path, ".json"))
        iftError("Invalid Output Json File: \"%s\"... Try (*.json)", "iftValidateRequiredArgs", out_disp_path);

    char *parent_dir = iftParentDir(out_disp_path);
    if (!iftDirExists(parent_dir)) {
        iftMakeDir(parent_dir);
    }
    iftFree(parent_dir);
}


void iftGetOptionalArgs(  iftDict *args, int *n_scales, int *stride) {
    if (iftDictContainKey("--num-of-scales", args, NULL))
        *n_scales = iftGetLongValFromDict("--num-of-scales", args);
    if (iftDictContainKey("--stride", args, NULL))
        *stride   = iftGetLongValFromDict("--stride", args);

    if (*n_scales <= 0)
        iftError("Invalid Number of Scales: %d <= 0", "iftGetOptionalArgs", *n_scales);
    if (*stride <= 0)
        iftError("Invalid Stride: %d <= 0", "iftGetOptionalArgs", *stride);


    printf("- Number of Scales: %d\n", *n_scales);
    printf("- Stride: %d\n", *stride);
    puts("-----------------------");
}


void iftWriteInfoFile(const char *fixed_grad_path, const char *mov_grad_path, iftVector best_disp,
                      const char *out_disp_path) {
    iftJson *info = iftCreateJsonRoot();

    iftAddStringToJson(info, "fixed-gradient-image", fixed_grad_path);
    iftAddStringToJson(info, "moving-gradient-image", mov_grad_path);
    iftAddJDictReferenceToJson(info, "best-disp", iftCreateJDict());
    iftAddDoubleToJson(info, "best-disp:x", best_disp.x);
    iftAddDoubleToJson(info, "best-disp:y", best_disp.y);
    iftAddDoubleToJson(info, "best-disp:z", best_disp.z);

    iftWriteJson(info, out_disp_path);

    iftDestroyJson(&info);
}




