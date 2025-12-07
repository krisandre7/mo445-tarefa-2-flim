#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **reg_mov_img_path, char **fix_img_path, char **fix_mask_path,
                        char **df, int *n_supervoxels, char **out_super_img_path);
void iftGetOptionalArgs(  iftDict *args, float *alpha, float *beta, char **normal_asymmap_path,
                        char **out_asymmap_path, float *dist_to_surface, int *min_super_vol);
iftImage *iftNativeBrainAsymMap(  iftImage *fix_img,   iftImage *reg_mov_img,   iftImage *normal_asymm_map);
iftMImage *iftBuildBrainNativeMImage(  iftImage *fix_img,   iftImage *reg_mov_img);
iftImage *iftNativeSymmISF(  iftImage *reg_mov_img,   iftImage *fix_img,   iftImage *fix_mask,
                           int n_supervoxels, float alpha, float beta,   iftImage *normal_asymmap,
                           const char *df_path, float dist_to_surface, int min_super_vol, iftImage **asym_map_out);
iftImage *iftRemoveSurfaceSupervoxels(  iftImage *super_side_img,   iftImage *fixed_mask, float dist_to_surface,
                                      int min_super_vol);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);
    
    char *reg_mov_img_path = NULL;
    char *fix_img_path = NULL;
    char *fix_mask_path = NULL;
    char *df = NULL;
    int n_supervoxels;
    char *out_super_img_path = NULL;
    // optional
    float alpha;
    float beta;
    char *normal_asymmap_path = NULL;
    char *out_asymmap_path = NULL;
    float dist_to_surface;
    int min_super_vol;
    
    iftGetRequiredArgs(args, &reg_mov_img_path, &fix_img_path, &fix_mask_path, &df, &n_supervoxels, &out_super_img_path);
    iftGetOptionalArgs(args, &alpha, &beta, &normal_asymmap_path, &out_asymmap_path, &dist_to_surface, &min_super_vol);

    timer *t1 = iftTic();
    
    iftImage *reg_mov_img = iftReadImageByExt(reg_mov_img_path);
    iftImage *fix_img = iftReadImageByExt(fix_img_path);
    iftImage *fix_mask = iftReadImageByExt(fix_mask_path);
    iftImage *normal_asymmap = (normal_asymmap_path) ? iftReadImageByExt(normal_asymmap_path) : NULL;
    
    puts("- Running Native SymmISF");
    iftImage *asym_map = NULL;
    iftImage *super_img = iftNativeSymmISF(reg_mov_img, fix_img, fix_mask, n_supervoxels, alpha, beta, normal_asymmap,
                                           df, dist_to_surface, min_super_vol, &asym_map);
    iftWriteImageByExt(super_img, out_super_img_path);
    iftWriteImageByExt(asym_map, out_asymmap_path);


    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyImage(&reg_mov_img);
    iftDestroyImage(&fix_img);
    iftDestroyImage(&fix_mask);
    iftDestroyImage(&normal_asymmap);
    iftDestroyImage(&super_img);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Run SymmISF on Native Space.\n" \
        "- Given a image masked with the fixed object and another image masked with the moving object after registration " \
        "to the first image, it extracts symmetric supervoxels to them." \
        "- An example is: an image with only the right hemisphere (fixed object), and another one with only the left " \
        "hemisphere (moving image) after flipping on sagittal plane and registration with the first image.\n" \
        "- The resulting symmetrical supervoxels is a mask with supervoxels only inside the fixed object. To also get the opposite supervoxels, " \
        "pass the corresponding inverse deformation fields to map the fixed object to the moving one.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-m", .long_name = "--registered-moving-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the Cropped/masked registered Image with the moving object."},
        {.short_name = "-f", .long_name = "--fixed-object-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the Cropped/masked Image with the fixed object."},
        {.short_name = "-l", .long_name = "--fixed-mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the mask with the fixed object."},
        {.short_name = "-r", .long_name = "--def-fields-file", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Inverse Deformation field to map the Fixed Object to the Moving one."},
        {.short_name = "-n", .long_name = "--number-of-supervoxels", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=true, .help="Required number of supervoxels."},
        {.short_name = "-o", .long_name = "--output-supervoxels-path", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the output symmetric supervoxel image."},
        {.short_name = "-a", .long_name = "--alpha", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Alpha factor of SymmISF. Default: 0.08"},
        {.short_name = "-b", .long_name = "--beta", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Beta factor of SymmISF. Default: 3"},
        {.short_name = "-s", .long_name = "--normal-asymmetry-map", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Pathname from the normal asymmetry map used to attenuate the brain asymmetries " \
                               "for the seed initialization for SymmISF."},
        {.short_name = "-t", .long_name = "--output-asymmetry-map", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Pathname to save the resulting asymmetry map."},
        {.short_name = "-d", .long_name = "--eucl-dist-to-surface", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Euclidean distance between supervoxel's geometric center and object's surface used " \
                                "to filter/remove supervoxels. (Try 5.0)"},
        {.short_name = "-v", .long_name = "--min-supervoxel-volume", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=false, .help="Minimum volume that supervoxels near from the object's surface must have (Try 400)."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **reg_mov_img_path, char **fix_img_path, char **fix_mask_path,
                        char **df, int *n_supervoxels, char **out_super_img_path) {
    *reg_mov_img_path = iftGetStrValFromDict("--registered-moving-image", args);
    *fix_img_path = iftGetStrValFromDict("--fixed-object-image", args);
    *fix_mask_path = iftGetStrValFromDict("--fixed-mask", args);
    *df = iftGetStrValFromDict("--def-fields-file", args);
    *n_supervoxels = iftGetLongValFromDict("--number-of-supervoxels", args);
    *out_super_img_path = iftGetStrValFromDict("--output-supervoxels-path", args);
    
    puts("-------------------------");
    printf("- Test Reg. Moving Image: %s\n", *reg_mov_img_path);
    printf("- Test Fixed Image: %s\n", *fix_img_path);
    printf("- Test Fixed Mask: %s\n", *fix_mask_path);
    printf("- Deformation Field: %s\n", *df);
    printf("- Number of Supervoxels: %d\n", *n_supervoxels);
    printf("- Output Supervoxels Image: %s\n", *out_super_img_path);
    puts("-------------------------");
}


void iftGetOptionalArgs(  iftDict *args, float *alpha, float *beta, char **normal_asymmap_path,
                        char **out_asymmap_path, float *dist_to_surface, int *min_super_vol) {
    if (iftDictContainKey("--alpha", args, NULL))
        *alpha = iftGetDblValFromDict("--alpha", args);
    else *alpha = 0.08;
    
    if (iftDictContainKey("--beta", args, NULL))
        *beta = iftGetDblValFromDict("--beta", args);
    else *beta = 3;
    
    if (iftDictContainKey("--normal-asymmetry-map", args, NULL))
        *normal_asymmap_path = iftGetStrValFromDict("--normal-asymmetry-map", args);
    else *normal_asymmap_path = NULL;
    
    if (iftDictContainKey("--output-asymmetry-map", args, NULL))
        *out_asymmap_path = iftGetStrValFromDict("--output-asymmetry-map", args);
    else *out_asymmap_path = NULL;
    
    if (iftDictContainKey("--eucl-dist-to-surface", args, NULL))
        *dist_to_surface = iftGetDblValFromDict("--eucl-dist-to-surface", args);
    else *dist_to_surface = -1;
    
    if (iftDictContainKey("--min-supervoxel-volume", args, NULL))
        *min_super_vol = iftGetLongValFromDict("--min-supervoxel-volume", args);
    else *min_super_vol = -1;
    
    if (*dist_to_surface < 0 && *min_super_vol >= 0)
        iftError("Eucl. Distance to Surface not passed, but Min. Supervoxel Volumes was passed. Pass both parameters",
                 "iftGetOptionalArgs");
    else if (*dist_to_surface >= 0 && *min_super_vol < 0)
        iftError("Eucl. Distance to Surface was passed, but Min. Supervoxel Volumes not. Pass both parameters",
                 "iftGetOptionalArgs");
    
    printf("- Alpha Factor: %f\n", *alpha);
    printf("- Beta Factor: %f\n", *beta);
    if (*normal_asymmap_path)
        printf("- Normal asymmetry Map: %s\n", *normal_asymmap_path);
    if (*out_asymmap_path)
        printf("- Output Asymmetry Map: %s\n", *out_asymmap_path);
    printf("- Eucl. Distance to Surface: %f\n", *dist_to_surface);
    printf("- Min. Supervoxel Volume: %d\n", *min_super_vol);
    puts("-------------------------\n");
}


iftImage *iftNativeBrainAsymMap(  iftImage *fix_img,   iftImage *reg_mov_img,   iftImage *normal_asymm_map) {
    iftVerifyImages(fix_img, reg_mov_img, "iftNativeBrainAsymMap");
    
    iftImage *diff = NULL;
    if (normal_asymm_map) {
        iftVerifyImages(fix_img, normal_asymm_map, "iftNativeBrainAsymMap");
        diff = iftCreateImageFromImage(fix_img);

        #pragma omp parallel for
        for (int p = 0; p < diff->n; p++)
            diff->val[p] = iftMax(0, abs(fix_img->val[p] - reg_mov_img->val[p]) - normal_asymm_map->val[p]);
    }
    else diff = iftAbsSub(fix_img, reg_mov_img);
    
    return diff;
}


iftMImage *iftBuildBrainNativeMImage(  iftImage *fix_img,   iftImage *reg_mov_img) {
    iftMImage *mimg = iftCreateMImage(fix_img->xsize, fix_img->ysize, fix_img->zsize, 2);

    #pragma omp parallel for
    for (int p = 0; p < mimg->n; p++) {
        mimg->val[p][0] = fix_img->val[p];
        mimg->val[p][1] = reg_mov_img->val[p];
    }
    
    return mimg;
}


iftImage *iftNativeSymmISF(  iftImage *reg_mov_img,   iftImage *fix_img,   iftImage *fix_mask,
                           int n_supervoxels, float alpha, float beta,   iftImage *normal_asymmap,
                           const char *df_path, float dist_to_surface, int min_super_vol, iftImage **asym_map_out) {
    iftImage *asym_map = iftNativeBrainAsymMap(fix_img, reg_mov_img, normal_asymmap);
    
    iftIntArray *grid = iftGridSamplingByBrainAsymmetries(asym_map, fix_mask, 30);
    iftImage *seeds = iftCreateImageFromImage(fix_img);
    iftIntArrayToImage(grid, seeds, 1);
    
    iftMImage *mimg = iftBuildBrainNativeMImage(fix_img, reg_mov_img);
    
    iftAdjRel *A = iftSpheric(1.0);
    iftIGraph *igraph = iftExplicitIGraph(mimg, fix_mask, NULL, A);
    iftIGraphISF_Root(igraph, seeds, alpha, beta, 10);
    
    iftImage *super_img = iftIGraphLabel(igraph);
    iftCopyVoxelSize(reg_mov_img, super_img);
    
    if (dist_to_surface >= 0 && min_super_vol >= 0) {
        puts("\t- Filtering Supervoxels");
        iftImage *filt_super_img = iftRemoveSurfaceSupervoxels(super_img, fix_mask, dist_to_surface, min_super_vol);
        iftDestroyImage(&super_img);
        super_img = filt_super_img;
    }
    
    if (df_path) {
        iftImage *super_img_reg = iftTransformImageByTransformix(super_img, df_path);
        iftImage *super_img_reg_flip = iftFlipImage(super_img_reg, IFT_AXIS_X);
        iftDestroyImage(&super_img_reg);

        #pragma omp parallel for
        for (int p = 0; p < super_img_reg_flip->n; p++)
            if (super_img->val[p] == 0)
                super_img->val[p] = super_img_reg_flip->val[p];
        
        iftDestroyImage(&super_img_reg_flip);
    }
    
    if (asym_map_out)
        *asym_map_out = asym_map;
    else iftDestroyImage(&asym_map);
    
    iftDestroyIntArray(&grid);
    iftDestroyImage(&seeds);
    iftDestroyMImage(&mimg);
    iftDestroyAdjRel(&A);
    iftDestroyIGraph(&igraph);
    
    return super_img;
}


iftImage *iftRemoveSurfaceSupervoxels(  iftImage *super_side_img,   iftImage *fixed_mask, float dist_to_surface,
                                      int min_super_vol) {
    float square_dist = dist_to_surface * dist_to_surface;
    iftImage *edt = iftEuclDistTrans(fixed_mask, NULL, IFT_BOTH, NULL, NULL, NULL);
    
    iftIntArray *n_voxels = iftCountLabelSpels(super_side_img);
    iftVoxelArray *gcs_arr = iftGeometricCenterVoxelsLabelImage(super_side_img);
    long n_labels = n_voxels->n - 1;
    
    // filters the supervoxels near from the object's surface with number of voxels less than the minimum required
    // it finds a mapping between the old supervoxels' labels to the new ones (after filtering)
    iftIntArray *new_labels = iftCreateIntArray(n_labels + 1);
    int last_valid_label = 0;
    
    for (long label = 1; label <= n_labels; label++) {
        iftVoxel gc = gcs_arr->val[label];
    
        if ((iftImgVoxelVal(edt, gc) <= square_dist) && (n_voxels->val[label] < min_super_vol)) {
            new_labels->val[label] = 0;
        } else {
            last_valid_label++;
            new_labels->val[label] = last_valid_label;
        }
    }
    
    iftImage *filt_super_side_img = iftCreateImageFromImage(super_side_img);

    #pragma omp parallel for
    for (int p = 0; p < super_side_img->n; p++) {
        int old_label = super_side_img->val[p];
        int new_label = new_labels->val[old_label];
        filt_super_side_img->val[p] = new_label;
    }
    
    iftDestroyImage(&edt);
    iftDestroyIntArray(&n_voxels);
    iftDestroyVoxelArray(&gcs_arr);
    
    return filt_super_side_img;
}
/*************************************************************/


