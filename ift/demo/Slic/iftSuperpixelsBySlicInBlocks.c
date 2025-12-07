/**
 * @file
 * @brief Divides the Input Image into blocks, generates superpixels for each block, and then re-enumerates
 * all block's superpixels, getting the final Superpixel Image.
 * @note See the source code in @ref iftSuperpixelsBySlicInBlocks.c
 *
 * @example iftSuperpixelsBySlicInBlocks.c
 * @brief Divides the Input Image into blocks, generates superpixels for each block, and then re-enumerates
 * all block's superpixels, getting the final Superpixel Image.
 * @author Samuel Martins
 * @date Jun 17, 2016
 */


#include "ift.h"

/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, int *input_n_blocks,
                        int *input_n_clusters, double *comp, char **out_img_path);
void iftValidateInputs(const char *img_path, int input_n_blocks, int input_n_clusters,
                       double comp, const char *out_img_path);
void iftGetOptionalArgs(  iftDict *args,   iftImage *img, int *img_max_range);
/*************************************************************/


int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    char *img_path;
    int input_n_blocks;
    int input_n_clusters;
    double comp;
    char *out_img_path;
    int img_max_range;

    iftGetRequiredArgs(args, &img_path, &input_n_blocks, &input_n_clusters, &comp, &out_img_path);

    puts("- Reading Input Image");
    iftImage *img = iftReadImageByExt(img_path);
    iftGetOptionalArgs(args, img, &img_max_range);

    
    timer *t1 = iftTic();
    puts("- Generating the Superpixel Image in Blocks");
    int n_clusters;
    iftImage *super_img = iftGenerateSuperpixelsBySlicInBlocks(img, NULL, input_n_blocks, input_n_clusters,
                                                                 comp, img_max_range, NULL, &n_clusters);
    printf("--> Number of Superpixels generated: %d\n", n_clusters);
    iftWriteImageByExt(super_img, out_img_path);

    if (iftDictContainKey("--save-border-image", args, NULL)) {
        char *filename        = iftFilename(out_img_path, iftFileExt(out_img_path));
        char *border_img_path = NULL;

        if (iftIs3DImage(img))
            border_img_path = iftConcatStrings(2, filename, "_borders.zip");
        else
            border_img_path = iftConcatStrings(2, filename, "_borders.png");

        iftWriteSuperpixelBorders(img, super_img, border_img_path);

        iftFree(filename);
        iftFree(border_img_path);
    }
    
    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));


    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyImage(&img);
    iftDestroyImage(&super_img);

    return 0;
}



/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "Divides the Input Image into blocks, generates superpixels for each block, and then re-enumerates " \
         "all block's superpixels, getting the final Superpixel Image.";


    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--input-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Input Image, which can be 2D or 3D, and grayscale or color image."},
        {.short_name = "-n", .long_name = "--num-blocks", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=true, .help="Expected Number of Blocks."},
        {.short_name = "-k", .long_name = "--num-clusters", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=true, .help="Approximate Number of Superpixels per Block."},
        {.short_name = "-c", .long_name = "--compactness", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=true, .help="Compactness: Relative Importance of color similarity and spatial proximity.\n" \
                                "Try values on a log scale (0.01, 0.1, 1, 10, 100) before refining " \
                                "around a chosen value."},
        {.short_name = "-o", .long_name = "--output-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Integer mask indicating the superpixels/supervoxels (*.pgm for 2D and *.scn for 3D)."},
        {.short_name = "-b", .long_name = "--img-depth", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=false, .help="Input Image Depth in bits (8, 12, 16, ...)\n" \
                                "Default: it tries to find the image depth automatically"},
        {.short_name = "-s", .long_name = "--save-border-image", .has_arg=false,
         .required=false, .help="Save the input Image with the Border of the Superpixels over it"}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **img_path, int *input_n_blocks,
                        int *input_n_clusters, double *comp, char **out_img_path) {
    *img_path         = iftGetStrValFromDict("--input-img", args);
    *input_n_blocks   = iftGetLongValFromDict("--num-blocks", args);
    *input_n_clusters = iftGetLongValFromDict("--num-clusters", args);
    *comp             = iftGetDblValFromDict("--compactness", args);
    *out_img_path     = iftGetStrValFromDict("--output-img", args);

    iftValidateInputs(*img_path, *input_n_blocks, *input_n_clusters, *comp, *out_img_path);

    puts("-----------------------");
    printf("- Image: \"%s\"\n", *img_path);
    printf("- Expected Number of Blocks: %d\n", *input_n_blocks);
    printf("- Input Number of Superpixels/Supervoxels: %d\n", *input_n_clusters);
    printf("- Compactness: %lf\n", *comp);
    printf("- Output Superpixel Image: \"%s\"\n", *out_img_path);
    puts("-----------------------");
}


void iftValidateInputs(const char *img_path, int input_n_blocks, int input_n_clusters,
                       double comp, const char *out_img_path) {
    if (!iftIsImageFile(img_path))
        iftError("Invalid Image: \"%s\"\n", "iftValidateInputs", img_path);

    if (input_n_blocks <= 0)
        iftError("Invalid Input Desired Number of Blocks: %d... Try > 0\n", "iftValidateInputs", input_n_blocks);

    if (input_n_clusters <= 0)
        iftError("Invalid Input Number of Superpixels: %ld... Try > 0", "iftValidateInputs", input_n_clusters);

    if (comp <= 0.0)
        iftError("Invalid Compactness: %lf... Try > 0 (0.01, 0.1, 1, 10)", "iftValidateInputs", comp);

    if (!iftIsImagePathnameValid(out_img_path))
        iftError("Invalid Output Image: \"%s\"\nTry *.[pgm, ppm, png, scn]", "iftValidateInputs",
                 out_img_path);

    char *parent_dir = iftParentDir(out_img_path);
    if (!iftDirExists(parent_dir)) {
        iftMakeDir(parent_dir);
    }
    iftFree(parent_dir);
}


void iftGetOptionalArgs(  iftDict *args,   iftImage *img, int *img_max_range) {
    if (iftDictContainKey("--img-depth", args, NULL)) {
        int img_depth = iftGetLongValFromDict("--img-depth", args);
        if (img_depth <= 0)
            iftError("Invalid Image Depth: %d... Try > 0", "iftGetOptionalArgs", img_depth);

        *img_max_range = (1 << img_depth) - 1; // (2^img_depth) - 1
    }
    else *img_max_range = iftNormalizationValue(iftMaximumValue(img));

    printf("- Image Range: [0, %d]\n", *img_max_range);
    puts("-----------------------\n");
}
/*************************************************************/










