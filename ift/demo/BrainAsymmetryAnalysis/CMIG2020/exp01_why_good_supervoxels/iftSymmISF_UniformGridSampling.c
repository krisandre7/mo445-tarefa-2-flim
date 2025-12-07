#include "ift.h"
#include "ift/medical/brain/AsymmetryAnalysis.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, char **bin_mask_path,
                        int *n_supervoxels, char **out_supervoxels_path);
void iftGetOptionalArgs(  iftDict *args, float *alpha, float *beta);

iftImage *iftSymmISF_GridSampling(  iftImage *img,   iftImage *bin_mask, int n_supervoxels, float alpha, float beta);
/*************************************************************/


int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);
    
    iftRandomSeed(IFT_RANDOM_SEED);
    
    // required args
    char *img_path = NULL;
    char *bin_mask_path = NULL;
    int n_supervoxels;
    char *out_supervoxels_path = NULL;
    // optional args
    float alpha;
    float beta;
    
    iftGetRequiredArgs(args, &img_path, &bin_mask_path, &n_supervoxels, &out_supervoxels_path);
    iftGetOptionalArgs(args, &alpha, &beta);
    
    timer *t1 = iftTic();
    
    
    iftImage *img = iftReadImageByExt(img_path);
    iftImage *bin_mask = iftReadImageByExt(bin_mask_path);
    
    puts("- Extracting Supervoxels");
    iftImage *super_img = iftSymmISF_GridSampling(img, bin_mask, n_supervoxels, alpha, beta);
    iftWriteImageByExt(super_img, out_supervoxels_path);
    
    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    
    
    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyImage(&img);
    iftDestroyImage(&bin_mask);
    iftDestroyImage(&super_img);
    
    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Extract symmetrical supervoxels on the Brain by SymmISF with uniform seed sampling.\n" \
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
            {.short_name = "-n", .long_name = "--number-of-supervoxels", .has_arg=true, .arg_type=IFT_LONG_TYPE,
             .required=true, .help="Required number of supervoxels."},
            {.short_name = "-o", .long_name = "--output-supervoxel-image-path", .has_arg=true, .arg_type=IFT_STR_TYPE,
             .required=true, .help="Output Symmetrical Supervoxels Image."},
            {.short_name = "-a", .long_name = "--alpha", .has_arg=true, .arg_type=IFT_DBL_TYPE,
             .required=false, .help="Alpha factor of SymmISF. Default: 0.08"},
            {.short_name = "-b", .long_name = "--beta", .has_arg=true, .arg_type=IFT_DBL_TYPE,
             .required=false, .help="Beta factor of SymmISF. Default: 3"}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);
    
    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);
    
    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **img_path, char **bin_mask_path,
                        int *n_supervoxels, char **out_supervoxels_path) {
    *img_path = iftGetStrValFromDict("--image-path", args);
    *bin_mask_path = iftGetStrValFromDict("--binary-mask", args);
    *n_supervoxels = iftGetLongValFromDict("--number-of-supervoxels", args);
    *out_supervoxels_path = iftGetStrValFromDict("--output-supervoxel-image-path", args);
    
    
    puts("-----------------------");
    printf("- Image: %s\n", *img_path);
    printf("- Binary Mask: %s\n", *bin_mask_path);
    printf("- Number of Supervoxels: %d\n", *n_supervoxels);
    printf("- Output Supervoxel Image: %s\n", *out_supervoxels_path);
    puts("-----------------------");
}


void iftGetOptionalArgs(  iftDict *args, float *alpha, float *beta) {
    if (iftDictContainKey("--alpha", args, NULL))
        *alpha = iftGetDblValFromDict("--alpha", args);
    else *alpha = 0.08;
    
    if (iftDictContainKey("--beta", args, NULL))
        *beta = iftGetDblValFromDict("--beta", args);
    else *beta = 3;
    
    printf("- Alpha Factor: %f\n", *alpha);
    printf("- Beta Factor: %f\n", *beta);
    puts("-----------------------\n");
}



iftImage *iftSymmISF_GridSampling(  iftImage *img,   iftImage *bin_mask, int n_supervoxels, float alpha, float beta) {
    float radius = iftEstimateGridOnMaskSamplingRadius(bin_mask, -1, n_supervoxels);
    iftIntArray* grid = iftGridSamplingOnMask(bin_mask, radius, -1, n_supervoxels);
    
    iftImage *seeds = iftCreateImageFromImage(img);
    iftIntArrayToImage(grid, seeds, 1);
    iftMImage *mimg = iftBuildBrainHemisphereMImage(img);
    
    iftAdjRel *A = iftSpheric(1.0);
    
    iftIGraph *igraph = iftExplicitIGraph(mimg, bin_mask, NULL, A);
    iftIGraphISF_Root(igraph, seeds, alpha, beta, 10);
    
    iftImage *supervoxels_img = iftIGraphLabel(igraph);
    
    int sagittal_midplane = img->xsize / 2;
    
    iftImage *final_svoxels_img = iftCreateImageFromImage(supervoxels_img);
    
    #pragma omp parallel for
    for (int p = 0; p < supervoxels_img->n; p++) {
        if (supervoxels_img->val[p]) {
            iftVoxel u = iftGetVoxelCoord(supervoxels_img, p);
            
            int disp_x = sagittal_midplane - u.x;
            iftVoxel v = {.x = sagittal_midplane + disp_x, .y = u.y, .z = u.z};
            
            iftImgVoxelVal(final_svoxels_img, u) = iftImgVoxelVal(supervoxels_img, u);
            iftImgVoxelVal(final_svoxels_img, v) = iftImgVoxelVal(supervoxels_img, u);
        }
    }
    
    iftDestroyIntArray(&grid);
    iftDestroyImage(&seeds);
    iftDestroyMImage(&mimg);
    iftDestroyAdjRel(&A);
    iftDestroyIGraph(&igraph);
    iftDestroyImage(&supervoxels_img);
    
    return final_svoxels_img;
}
/*************************************************************/


