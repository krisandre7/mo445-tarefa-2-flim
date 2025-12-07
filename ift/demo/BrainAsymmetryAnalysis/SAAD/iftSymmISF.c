#include "ift.h"
#include "ift/medical/brain/AsymmetryAnalysis.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, char **bin_mask_path,
                        char **out_supervoxels_path);
void iftGetOptionalArgs(  iftDict *args, float *alpha, float *beta, float *thres_factor, float *min_dist,
                        int *n_seeds_on_symmetric_regions, char **normal_asymmap_path);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    iftRandomSeed(IFT_RANDOM_SEED);
    
    // required args
    char *img_path = NULL;
    char *bin_mask_path = NULL;
    char *out_supervoxels_path = NULL;
    // optional args
    float alpha;
    float beta;
    float thres_factor;
    float min_dist;
    int n_seeds_on_symmetric_regions;
    char *normal_asymmap_path = NULL;

    iftGetRequiredArgs(args, &img_path, &bin_mask_path, &out_supervoxels_path);
    iftGetOptionalArgs(args, &alpha, &beta, &thres_factor, &min_dist, &n_seeds_on_symmetric_regions, &normal_asymmap_path);

    timer *t1 = iftTic();


    iftImage *img = iftReadImageByExt(img_path);
    iftImage *bin_mask = iftReadImageByExt(bin_mask_path);
    iftImage *normal_asymmap = (normal_asymmap_path) ? iftReadImageByExt(normal_asymmap_path) : NULL;
    
    puts("- Extracting Supervoxels");
    iftImage *super_img = iftSymmISF(img, bin_mask, alpha, beta, thres_factor, min_dist, n_seeds_on_symmetric_regions,
                                     normal_asymmap);
    iftWriteImageByExt(super_img, out_supervoxels_path);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyImage(&img);
    iftDestroyImage(&bin_mask);
    iftDestroyImage(&normal_asymmap);
    iftDestroyImage(&super_img);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Extract symmetrical supervoxels on the Brain by SymmISF.\n" \
        "- It expects a binary mask with the target object from a brain side (according to its Mid-Sagittal Plane), e.g " \
        "a mask with right hemisphere.\n" \
        "- Then, it performs the SymmISF inside this mask to generate the supervoxels, so that " \
        "the same supervoxels are mirrored in the other brain side.\n" \
        "- The parameters used by SymmISF can be passed.\n" \
        "- Small Supervoxels near from the object' surface whose volume is less than a given threshold (optionally passed) can be removed.\n\n" \
        "- PS: It assumes that the MSP is the central sagittal slice (xslice) from the image.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--image-path", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Image pathname."},
        {.short_name = "-m", .long_name = "--binary-mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Binary Mask with the target object from a brain side."},
        {.short_name = "-o", .long_name = "--output-supervoxel-image-path", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Symmetrical Supervoxels Image."},
        {.short_name = "-a", .long_name = "--alpha", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Alpha factor of SymmISF. Default: 0.08"},
        {.short_name = "-b", .long_name = "--beta", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Beta factor of SymmISF. Default: 3"},
        {.short_name = "-f", .long_name = "--threshold-otsu-factor", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Factor used to increase/decrease on otsu threshold to binarize the map. Default: 2.0"},
        {.short_name = "-d", .long_name = "--min-euclidean-distance", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Minimum euclidean distance to the binary mask borders that the initial seeds must have. " \
                                "Default: 0.0 (no restriction)."},
        {.short_name = "-y", .long_name = "--number-seeds-on-symmetric-regions", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                .required=false, .help="Number of Seeds on Symmetric Regions. Default: 50."},
        {.short_name = "-s", .long_name = "--normal-asymmetry-map", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Pathname from the normal asymmetry map used to attenuate the brain asymmetries " \
                                "for the seed initialization for SymmISF."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **img_path, char **bin_mask_path,
                        char **out_supervoxels_path) {
    *img_path = iftGetStrValFromDict("--image-path", args);
    *bin_mask_path = iftGetStrValFromDict("--binary-mask", args);
    *out_supervoxels_path = iftGetStrValFromDict("--output-supervoxel-image-path", args);


    puts("-----------------------");
    printf("- Image: %s\n", *img_path);
    printf("- Binary Mask: %s\n", *bin_mask_path);
    printf("- Output Supervoxel Image: %s\n", *out_supervoxels_path);
    puts("-----------------------");
}


void iftGetOptionalArgs(  iftDict *args, float *alpha, float *beta, float *thres_factor, float *min_dist,
                        int *n_seeds_on_symmetric_regions, char **normal_asymmap_path) {
    if (iftDictContainKey("--alpha", args, NULL))
        *alpha = iftGetDblValFromDict("--alpha", args);
    else *alpha = 0.08;

    if (iftDictContainKey("--beta", args, NULL))
        *beta = iftGetDblValFromDict("--beta", args);
    else *beta = 3;
    
    if (iftDictContainKey("--threshold-otsu-factor", args, NULL))
        *thres_factor = iftGetDblValFromDict("--threshold-otsu-factor", args);
    else *thres_factor = 2.0;
    
    if (iftDictContainKey("--min-euclidean-distance", args, NULL))
        *min_dist = iftGetDblValFromDict("--min-euclidean-distance", args);
    else *min_dist = 0.0;
    
    if (iftDictContainKey("--number-seeds-on-symmetric-regions", args, NULL))
        *n_seeds_on_symmetric_regions = iftGetLongValFromDict("--number-seeds-on-symmetric-regions", args);
    else *n_seeds_on_symmetric_regions = 0.0;

    if (iftDictContainKey("--normal-asymmetry-map", args, NULL))
        *normal_asymmap_path = iftGetStrValFromDict("--normal-asymmetry-map", args);
    else *normal_asymmap_path = NULL;
    
    printf("- Alpha Factor: %f\n", *alpha);
    printf("- Beta Factor: %f\n", *beta);
    printf("- Threshold Otsu Factor: %f\n", *thres_factor);
    printf("- Min. Euclidean Distance from initial seeds to the Binary Mask Borders: %f\n", *min_dist);
    printf("- Number of Seeds on Symmetric Regions: %d\n", *n_seeds_on_symmetric_regions);
    if (*normal_asymmap_path)
        printf("- Normal asymmetry Map: %s\n", *normal_asymmap_path);
    puts("-----------------------\n");
}
/*************************************************************/


