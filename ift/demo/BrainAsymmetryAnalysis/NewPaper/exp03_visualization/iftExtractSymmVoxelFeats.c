#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, char **symm_svoxels_path,
                        char **gt_lesions_path, char **out_dataset_path);
void iftGetOptionalArgs(  iftDict *args, int *n_bins, float *importance_factor_hist_feats,
                        char **normal_asym_map_path, bool *include_voxel_coord_feats);
/*************************************************************/


iftDataSet *iftSupervoxelHAAFeatsInAsymMapSingleImage(  iftImage *asym_map,   iftImage *svoxels_brain_side,
                                                      int n_bins) {
    int n_svoxels = iftMaximumValue(svoxels_brain_side);
    iftDataSet *Z = iftCreateDataSet(n_svoxels + 1, n_bins);  // [i] = feats from supervoxel with label i
    
    iftHist **norm_hists = iftCalcGrayImageHistForLabels(asym_map, svoxels_brain_side, n_bins, true, NULL); // size n_svoxels + 1

    #pragma omp parallel for
    for (int label = 1; label <= n_svoxels; label++) {
        for (int f = 0; f < Z->nfeats; f++) {
            Z->sample[label].feat[f] = norm_hists[label]->val[f];
        }
        iftDestroyHist(&norm_hists[label]);
    }
    
    iftDestroyHist(&norm_hists[0]);
    iftFree(norm_hists);
    
    return Z;
}


iftDataSet *iftExtractSymmVoxelFeats(  iftImage *img,   iftImage *symm_svoxels,   iftImage *gt_lesions,
                                       int n_bins, float importance_factor_hist_feats,   iftImage *normal_asym_map,
                                     bool include_voxel_coord_feats, iftDataSet **Z_hist_feats_out) {
    iftImage *asym_map = iftBrainAsymMap(img, normal_asym_map);
    iftImage *svoxels_right_side = iftExtractBrainSide(symm_svoxels, IFT_RIGHT_BRAIN_SIDE);
    
    iftIntArray *hemispheric_voxels = iftGetObjectVoxels(svoxels_right_side);
    
    int n_feats = n_bins + (3 * include_voxel_coord_feats);
    
    // [s] = feats from the hemispheric voxel at coordinate [s]
    iftDataSet *Z = iftCreateDataSet(hemispheric_voxels->n, n_feats);
    
    // shape: (n_svoxels + 1, n_bins) ===> [i] = feats from supervoxel with label i
    iftDataSet *Z_hist_feats = iftSupervoxelHAAFeatsInAsymMapSingleImage(asym_map, svoxels_right_side, n_bins);
    
    // if include_voxel_coord_feats==True, all hist features together coorespond to 50% of the final feat. vector's importance
    float scale_factor_hist_feats = (include_voxel_coord_feats) ? (importance_factor_hist_feats / n_bins) : 1.0;
    printf("scale_factor_hist_feats = %f\n", scale_factor_hist_feats);
    int sagittal_midplane = svoxels_right_side->xsize / 2;
    
    #pragma omp parallel for
    for (int s = 0; s < hemispheric_voxels->n; s++) {
        int p = hemispheric_voxels->val[s];
        int label = svoxels_right_side->val[p];
        
        iftVoxel u = iftGetVoxelCoord(svoxels_right_side, p);
        // symmetric voxel
        int disp_x = sagittal_midplane - u.x;
        iftVoxel v = {.x = sagittal_midplane + disp_x, .y = u.y, .z = u.z};
        
        for (int f = 0; f < Z_hist_feats->nfeats; f++) {
            Z->sample[s].feat[f] = Z_hist_feats->sample[label].feat[f] * scale_factor_hist_feats;
            Z->sample[s].group = label;
            
            if (iftImgVoxelVal(gt_lesions, u))
                Z->sample[s].truelabel = iftImgVoxelVal(gt_lesions, u);
            else if (iftImgVoxelVal(gt_lesions, v))
                Z->sample[s].truelabel = iftImgVoxelVal(gt_lesions, v);
            else Z->sample[s].truelabel = 0;
        }
    }
    
    
    if (include_voxel_coord_feats) {
        float importance_factor_coord_feats = 1 - importance_factor_hist_feats;
        float scale_factor_coord_feats = importance_factor_coord_feats / 3;
        printf("scale_factor_coord_feats = %f\n", scale_factor_coord_feats);
        
        #pragma omp parallel for
        for (int s = 0; s < hemispheric_voxels->n; s++) {
            int p = hemispheric_voxels->val[s];
            iftVoxel u = iftGetVoxelCoord(svoxels_right_side, p);
    
            Z->sample[s].feat[n_bins]       = (u.x / (1.0 * (svoxels_right_side->xsize - 1))) * scale_factor_coord_feats;
            Z->sample[s].feat[n_bins + 1] = (u.y / (1.0 * (svoxels_right_side->ysize - 1))) * scale_factor_coord_feats;
            Z->sample[s].feat[n_bins + 2] = (u.z / (1.0 * (svoxels_right_side->zsize - 1))) * scale_factor_coord_feats;
        }
    }
    
    
    iftDestroyImage(&asym_map);
    iftDestroyImage(&svoxels_right_side);
    iftDestroyIntArray(&hemispheric_voxels);
    
    if (Z_hist_feats_out)
        *Z_hist_feats_out = Z_hist_feats;
    else iftDestroyDataSet(&Z_hist_feats);
    
    return Z;
}





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);
    
    char *img_path = NULL;
    char *symm_svoxels_path = NULL;
    char *gt_lesions_path = NULL;
    char *out_dataset_path = NULL;
    
    int n_bins;
    float importance_factor_hist_feats = 0.5;
    char *normal_asym_map_path = NULL;
    bool include_voxel_coord_feats;
    
    iftGetRequiredArgs(args, &img_path, &symm_svoxels_path, &gt_lesions_path, &out_dataset_path);
    iftGetOptionalArgs(args, &n_bins, &importance_factor_hist_feats, &normal_asym_map_path, &include_voxel_coord_feats);
    
    timer *t1 = iftTic();
    
    iftImage *img = iftReadImageByExt(img_path);
    iftImage *symm_svoxels = iftReadImageByExt(symm_svoxels_path);
    iftImage *gt_lesions = iftReadImageByExt(gt_lesions_path);
    iftImage *normal_asym_map = (normal_asym_map_path) ? iftReadImageByExt(normal_asym_map_path) : NULL;
    
    puts("- Extracting Symmetric Voxel Feats");
    iftDataSet *Z_hist_feats_out = NULL;
    iftDataSet *Z = iftExtractSymmVoxelFeats(img, symm_svoxels, gt_lesions, n_bins, importance_factor_hist_feats,
                                             normal_asym_map, include_voxel_coord_feats, &Z_hist_feats_out);
    printf("\t- (n_samples, n_feats) = (%d, %d)\n", Z->nsamples, Z->nfeats);
    puts("- Writing dataset");
    iftWriteDataSet(Z, out_dataset_path);
    iftWriteDataSet(Z_hist_feats_out, iftReplaceString(out_dataset_path, ".zip", "_svoxels_hist_feats.zip"));
    
    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Build a dataset with voxel feats by using SAAD features.\n" \
        "- For each voxel of a supervoxels inside right brain hemisphere, its feature vectors correspond to:\n" \
        "  (i) the feature vector of its symmetric supervoxels (i.e., normalized histogram of absolute asymmetries); and\n"
        "  (ii) the feature vector of (i) + the normalized x-, y-, and z-coordinates (location);\n"
        "       An importance factor f in [0, 1] must be passed to ponder the importance of the texture features (f) and " \
        "       location features (1 - f)";

    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-i", .long_name = "--image", .has_arg=true, .arg_type=IFT_STR_TYPE,
             .required=true, .help="Test Image used to build the supervoxel datasets."},
            {.short_name = "-s", .long_name = "--symmetric-supervoxels", .has_arg=true, .arg_type=IFT_STR_TYPE,
             .required=true, .help="Label Image with the symmetric supervoxels."},
            {.short_name = "-g", .long_name = "--ground-truth-lesions", .has_arg=true, .arg_type=IFT_STR_TYPE,
             .required=true, .help="Label Image with the ground lesion segmentations."},
            {.short_name = "-o", .long_name = "--output-dataset-path", .has_arg=true, .arg_type=IFT_STR_TYPE,
             .required=true, .help="Output dataset path (*.zip)."},
            {.short_name = "-n", .long_name = "--num-of-bins", .has_arg=true, .arg_type=IFT_LONG_TYPE,
             .required=false, .help="Number of bins of the histogram. Default: 128."},
            {.short_name = "-f", .long_name = "--importance-factor-texture-features", .has_arg=true, .arg_type=IFT_DBL_TYPE,
             .required=false, .help="An importance factor in [0, 1] for the histogram features. " \
                                     "Location features will have an importance factor of (1 - this_factor). Default: 0.5"},
            {.short_name = "-m", .long_name = "--normal-asymmetry-map", .has_arg=true, .arg_type=IFT_STR_TYPE,
             .required=false, .help="Normal asymmetry map used to attenuate the testing image asymmetries."},
            {.short_name = "", .long_name = "--include-voxel-coordinate-feats", .has_arg=false,
             .required=false, .help="Include Voxel Coordinate Features."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **img_path, char **symm_svoxels_path,
                        char **gt_lesions_path, char **out_dataset_path) {
    *img_path = iftGetStrValFromDict("--image", args);
    *symm_svoxels_path = iftGetStrValFromDict("--symmetric-supervoxels", args);
    *gt_lesions_path = iftGetStrValFromDict("--ground-truth-lesions", args);
    *out_dataset_path = iftGetStrValFromDict("--output-dataset-path", args);
    
    puts("--------------------------");
    printf("- Image: %s\n", *img_path);
    printf("- Symmetric Supervoxels: %s\n", *symm_svoxels_path);
    printf("- Ground Truth lesions: %s\n", *gt_lesions_path);
    printf("- Output Dataset: %s\n", *out_dataset_path);
    puts("--------------------------");
}


void iftGetOptionalArgs(  iftDict *args, int *n_bins, float *importance_factor_hist_feats,
                        char **normal_asym_map_path, bool *include_voxel_coord_feats) {
    if (iftDictContainKey("--num-of-bins", args, NULL))
        *n_bins = (int) iftGetLongValFromDict("--num-of-bins", args);
    else *n_bins = 128;
    printf("- Num of Bins: %d\n", *n_bins);
    
    if (iftDictContainKey("--importance-factor-texture-features", args, NULL)) {
        *importance_factor_hist_feats = iftGetDblValFromDict("--importance-factor-texture-features", args);
        
        if (*importance_factor_hist_feats < 0.0 || *importance_factor_hist_feats > 1.0)
            iftError("Invalid Importance Factor for Histogram Feats: %f. Try [0, 1]", "iftGetOptionalArgs",
                    *importance_factor_hist_feats);
    }
    else *importance_factor_hist_feats = 0.5;
    printf("- Importance Fator for (Texture) Histogram Feats: %f\n", *importance_factor_hist_feats);
    
    
    if (iftDictContainKey("--normal-asymmetry-map", args, NULL)) {
        *normal_asym_map_path = iftGetStrValFromDict("--normal-asymmetry-map", args);
        printf("- Normal Asymmetry Map: %s\n", *normal_asym_map_path);
    }
    else *normal_asym_map_path = NULL;
    
    *include_voxel_coord_feats = iftDictContainKey("--include-voxel-coordinate-feats", args, NULL);
    printf("- Include Voxel Coordinate Feats: %s\n", iftBoolAsString(*include_voxel_coord_feats));
    
    puts("--------------------------\n");
}
/*************************************************************/


