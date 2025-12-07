
#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, char **mask_path, char **out_path);
void iftGetOptionalArgs(  iftDict *args, float *radius, float *sensitivity);
iftImage *InhomogeneityCorrection(  iftImage *img,   iftImage *mask, float radius, float sensitivity);
/*************************************************************/


int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);
    
    // mandatory args
    char *img_path  = NULL;
    char *mask_path = NULL;
    char *out_path  = NULL;

    // optional args
    float radius;
    float sensitivity;

    iftGetRequiredArgs(args, &img_path, &mask_path, &out_path);
    iftGetOptionalArgs(args, &radius, &sensitivity);

    timer *t1 = iftTic();
    
    puts("- Reading Input Image");
    iftImage *img = iftReadImageByExt(img_path);

    puts("- Reading Mask");
    iftImage *mask = iftReadImageByExt(mask_path);

    puts("- Inhomogeneity Correction");
    iftImage *corr_img = InhomogeneityCorrection(img, mask, radius, sensitivity);

    puts("- Writing Corrected Image");
    iftWriteImageByExt(corr_img, out_path);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));

    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(img_path);
    iftFree(out_path);
    iftFree(mask_path);
    iftDestroyImage(&img);
    iftDestroyImage(&mask);
    iftDestroyImage(&corr_img);

    return 0;
}



/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Apply Bias Field Correction by Cappabianco's method [1]\n\n" \
        "[1] Segmentação de tecidos do cérebro humano em imagens de ressonância magnética e sua avaliação - PhD Thesis";


    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--input-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Input Image to be corrected."},
        {.short_name = "-m", .long_name = "--mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Binary mask (skull stripping) that defines the structure of your interest."},
        {.short_name = "-o", .long_name = "--output-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the output image corrected by N4."},
        {.short_name = "-r", .long_name = "--adjacency-radius", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Adjacency Radius (>= 1). Default: 18.3"},
        {.short_name = "-a", .long_name = "--sensitivy-factor", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Sensitivity Factor (>= 0). Default: 0.3"},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **img_path, char **mask_path, char **out_path) {
    *img_path  = iftGetStrValFromDict("--input-img", args);
    *mask_path = iftGetStrValFromDict("--mask", args);
    *out_path  = iftGetStrValFromDict("--output-img", args);

    if (!iftFileExists(*img_path))
        iftError("Input image %s does not exists", "iftGetRequiredArgs", *img_path);

    puts("-----------------------");
    printf("- Input Image: %s\n", *img_path);
    printf("- Brain Envelope Binary Mask: %s\n", *mask_path);
    printf("- Output Image: %s\n", *out_path);
    puts("-----------------------");
}


void iftGetOptionalArgs(  iftDict *args, float *radius, float *sensitivity) {
    if (iftDictContainKey("--adjacency-radius", args, NULL))
        *radius = iftGetDblValFromDict("--adjacency-radius", args);
    else *radius = 18.3;

    if (*radius < 1.0)
        iftError("Invalid Adjacency Radius: %f < 1.0", "iftGetOptionalArgs", *radius);

    if (iftDictContainKey("--sensitivy-factor", args, NULL))
        *sensitivity = iftGetDblValFromDict("--sensitivy-factor", args);
    else *sensitivity = 0.3;

    if (*sensitivity < 0.0)
        iftError("Sensitivity Factor: %f < 1.0", "iftGetOptionalArgs", *sensitivity);

    printf("- Radius: %f mm\n", *radius);
    printf("- Sensitivity Factor: %f\n", *sensitivity);
    puts("-----------------------\n");

}


iftImage *InhomogeneityCorrection(  iftImage *img,   iftImage *mask, float radius, float sensitivity) {
    iftImage *masked_img = iftMask(img, mask);
    int brain_max = iftMaximumValue(img);
    printf("brain_max = %d\n", brain_max);

    iftImage *corr_img = iftCopyImage(img);

    iftAdjRel *A      = iftSpheric(radius);
    iftIntArray *adjs = iftCreateIntArray(A->n);
    iftIntArray *idxs  = iftCreateIntArray(A->n);
    int n_relevant_voxels = 15;
    printf("A->n: %d\n", A->n);

    for (int p = 0; p < img->n; p++) {
        // if (mask->val[p] != 0) {
            iftVoxel u = iftGetVoxelCoord(mask, p);
            int pos = 0;

            for (int i = 1; i < A->n; i++) {
                iftVoxel v = iftGetAdjacentVoxel(A, u, i);

                if (iftValidVoxel(img, v)) {
                    adjs->val[pos] = iftImgVoxelVal(img, v);
                    idxs->val[pos] = iftGetVoxelIndex(img, v);
                    pos++;
                }
            }
            adjs->n = idxs->n = pos; // some adjacent voxels could be out of the image's domain
            iftQuickSort(adjs->val, idxs->val, 0, adjs->n-1, IFT_DECREASING);
            int median = adjs->val[n_relevant_voxels / 2];

            corr_img->val[p] = brain_max - ((median - img->val[p]) * powf((2 - (median / brain_max)), sensitivity));
            adjs->n = idxs->n = A->n; // reset array
        // }
    }


    iftDestroyAdjRel(&A);
    iftDestroyImage(&masked_img);
    iftDestroyIntArray(&adjs);
    iftDestroyIntArray(&idxs);

    return corr_img;
}
























