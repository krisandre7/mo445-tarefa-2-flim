#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, char **symm_svoxels_path,
                        char **gt_lesions_path, char **out_dataset_path);
void iftGetOptionalArgs(  iftDict *args, int *n_bins, char **normal_asym_map_path);
/*************************************************************/


iftDataSet *iftSymmetricSupervoxelHAAFeats(  iftImage *img,   iftImage *symm_svoxels,   iftImage *gt_lesions,
                                             int n_bins,   iftImage *normal_asym_map) {
    iftImage *asym_map = iftBrainAsymMap(img, normal_asym_map);
    iftImage *svoxels_right_side = iftExtractBrainSide(symm_svoxels, IFT_RIGHT_BRAIN_SIDE);
    
    iftImage *gt_lesions_bin = iftBinarize(gt_lesions);
    iftImage *svoxels_intersec_gts = iftMult(symm_svoxels, gt_lesions_bin);
    
    iftIntArray *svoxels_vols = iftCountLabelSpels(svoxels_right_side);
    iftIntArray *svoxels_intersec_gts_vols = iftCountLabelSpels(svoxels_intersec_gts);
    
    int n_svoxels = iftMaximumValue(svoxels_right_side);
    iftDataSet *Z = iftCreateDataSet(n_svoxels + 1, n_bins);  // [i] = feats from supervoxel with label i
    
    iftHist **norm_hists = iftCalcGrayImageHistForLabels(asym_map, svoxels_right_side, n_bins, true, NULL); // size n_svoxels + 1

    puts("perc_overlap");
    
    #pragma omp parallel for
    for (int label = 1; label <= n_svoxels; label++) {
        float perc_overlap = 0.0;
        if (label < svoxels_intersec_gts_vols->n) // svoxels_intersec_gts_vols->n = max_label_of_the_intersection + 1
            perc_overlap = (svoxels_intersec_gts_vols->val[label] * 1.0) / iftMax(svoxels_vols->val[label], 1.0);
        
        Z->sample[label].truelabel = (perc_overlap >= 0.1) ? 1 : 0;
        Z->sample[label].weight = perc_overlap;
        printf("label: %d ===> %f\n", label, perc_overlap);
        printf("Z->sample[%d].truelabel = %d\n\n", label, Z->sample[label].truelabel);
    
    
        for (int f = 0; f < Z->nfeats; f++) {
            Z->sample[label].feat[f] = norm_hists[label]->val[f];
        }
        iftDestroyHist(&norm_hists[label]);
    }
    iftDestroyHist(&norm_hists[0]);
    iftFree(norm_hists);
    
    iftDestroyImage(&asym_map);
    iftDestroyImage(&svoxels_right_side);
    iftDestroyImage(&gt_lesions_bin);
    iftDestroyImage(&svoxels_intersec_gts);
    iftDestroyIntArray(&svoxels_vols);
    iftDestroyIntArray(&svoxels_intersec_gts_vols);
    
    return Z;
}



int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);
    
    char *img_path = NULL;
    char *symm_svoxels_path = NULL;
    char *gt_lesions_path = NULL;
    char *out_dataset_path = NULL;
    
    int n_bins;
    char *normal_asym_map_path = NULL;
    
    iftGetRequiredArgs(args, &img_path, &symm_svoxels_path, &gt_lesions_path, &out_dataset_path);
    iftGetOptionalArgs(args, &n_bins, &normal_asym_map_path);
    
    timer *t1 = iftTic();
    
    iftImage *img = iftReadImageByExt(img_path);
    iftImage *symm_svoxels = iftReadImageByExt(symm_svoxels_path);
    iftImage *gt_lesions = iftReadImageByExt(gt_lesions_path);
    iftImage *normal_asym_map = (normal_asym_map_path) ? iftReadImageByExt(normal_asym_map_path) : NULL;
    
    puts("- Extracting Symmetric Voxel Feats");
    iftDataSet *Z_hist_feats_out = iftSymmetricSupervoxelHAAFeats(img, symm_svoxels, gt_lesions, n_bins, normal_asym_map);
    printf("\t- (n_samples, n_feats) = (%d, %d)\n", Z_hist_feats_out->nsamples, Z_hist_feats_out->nfeats);
    puts("- Writing dataset");
    iftWriteDataSet(Z_hist_feats_out, out_dataset_path);
    
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
        "- For each voxel of a supervoxels inside right brain hemisphere, its feature vectors may correspond to:\n" \
        "  (i) the feature vector of its symmetric supervoxels (i.e., normalized histogram of absolute asymmetries); and\n"
        "  (ii) the feature vector of (i) + the normalized x-, y-, and z-coordinates (location);\n"
        "       To avoid a biased vector, we ponder each feature so that the 3 location features have the same" \
        "       importance of the histogram feats.";

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
            {.short_name = "-m", .long_name = "--normal-asymmetry-map", .has_arg=true, .arg_type=IFT_STR_TYPE,
             .required=false, .help="Normal asymmetry map used to attenuate the testing image asymmetries."}
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


void iftGetOptionalArgs(  iftDict *args, int *n_bins, char **normal_asym_map_path) {
    if (iftDictContainKey("--num-of-bins", args, NULL))
        *n_bins = (int) iftGetLongValFromDict("--num-of-bins", args);
    else *n_bins = 128;
    printf("- Num of Bins: %d\n", *n_bins);
    
    if (iftDictContainKey("--normal-asymmetry-map", args, NULL)) {
        *normal_asym_map_path = iftGetStrValFromDict("--normal-asymmetry-map", args);
        printf("- Normal Asymmetry Map: %s\n", *normal_asym_map_path);
    }
    else *normal_asym_map_path = NULL;
    
    puts("--------------------------\n");
}
/*************************************************************/


