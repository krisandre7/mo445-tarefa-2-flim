#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **asymmap_path, char **fix_mask_path, char **symm_super_path,
                        iftFileSet **train_asymmap_set, char **out_dir);
void iftGetOptionalArgs(  iftDict *args, int *n_bins);
iftDataSet **iftExtractNativeSupervoxelHAAFeats(  iftImage *test_asymmap,   iftImage *fix_mask,   iftImage *symm_super_img,
                                                  iftFileSet *train_asymmap_set, int n_bins, int *n_supervoxels_out);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);
    
    char *asymmap_path = NULL;
    char *fix_mask_path = NULL;
    char *symm_super_path = NULL;
    iftFileSet *train_asymmap_set = NULL;
    char *out_dir = NULL;
    // optional args
    int n_bins;
    
    iftGetRequiredArgs(args, &asymmap_path, &fix_mask_path, &symm_super_path, &train_asymmap_set, &out_dir);
    iftGetOptionalArgs(args, &n_bins);

    timer *t1 = iftTic();

    iftImage *asymmap = iftReadImageByExt(asymmap_path);
    iftImage *fix_mask = iftReadImageByExt(fix_mask_path);
    iftImage *symm_super_img = iftReadImageByExt(symm_super_path);
    
    puts("- Extracting HAA Feats");
    int n_supervoxels;
    iftDataSet **Zarr = iftExtractNativeSupervoxelHAAFeats(asymmap, fix_mask, symm_super_img, train_asymmap_set, n_bins,
                                                           &n_supervoxels);
    puts("- Building Reference Data");
    iftBuildRefDataSupervoxelDataSets(Zarr, n_supervoxels, asymmap_path, symm_super_path, train_asymmap_set);
    
    
    puts("- Writing Supervoxel Datasets");
    char *out_datasets_dir = iftJoinPathnames(2, out_dir, iftCopyString("datasets_nbins_%d", n_bins));
    if (!iftDirExists(out_datasets_dir))
        iftMakeDir(out_datasets_dir);
    
    #pragma omp parallel for
    for (int label = 1; label <= n_supervoxels; label++) {
        char *filename = iftCopyString("supervoxel_%04d.zip", label);
        char *out_dataset_path = iftJoinPathnames(2, out_datasets_dir, filename);
        
        iftWriteDataSet(Zarr[label], out_dataset_path);
        
        iftFree(filename);
        iftFree(out_dataset_path);
        iftDestroyDataSet(&Zarr[label]);
    }
    iftFree(Zarr);


    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));
    

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyImage(&asymmap);
    iftDestroyImage(&fix_mask);
    iftDestroyImage(&symm_super_img);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Extracts the histogram of absolute asymmetries (HAA) between each pair of symmetric supervoxels of a given test image.\n" \
        "- For each pair of symmetric supervoxels, it creates a dataset where such pair is a sample.\n" \
        "- Each dataset has one sample extracted for the test image and one sample per training image from a training control set.\n" \
        "- A normal asymmetry map can be passed to attenuate the absolute asymmetries in the test image.\n\n" \
        "- The filename of each dataset is supervoxel_XXXX.zip, where XXXX is the supervoxel label, starting at 0001.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-a", .long_name = "--test-asymmetry-map", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname to save the test asymmetry map."},
        {.short_name = "-f", .long_name = "--fixed-mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the mask with the fixed object."},
        {.short_name = "-s", .long_name = "--symmetric-supervoxels-path", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the symmetric supervoxel image."},
        {.short_name = "-y", .long_name = "--train-asym-maps-on-native-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Dir. or CSV with the pathname from the Asymmetry maps from the training images."},
        {.short_name = "-o", .long_name = "--output-dir", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Directory where the symmetry supervoxel datasets will be saved."},
        {.short_name = "-n", .long_name = "--num-of-bins", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=false, .help="Number of bins of the histogram. Default: 128."},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **asymmap_path, char **fix_mask_path, char **symm_super_path,
                        iftFileSet **train_asymmap_set, char **out_dir) {
    *asymmap_path = iftGetStrValFromDict("--test-asymmetry-map", args);
    *fix_mask_path = iftGetStrValFromDict("--fixed-mask", args);
    *symm_super_path = iftGetStrValFromDict("--symmetric-supervoxels-path", args);
    const char *train_asymmap_set_entry = iftGetConstStrValFromDict("--train-asym-maps-on-native-entry", args);
    *train_asymmap_set = iftLoadFileSetFromDirOrCSV(train_asymmap_set_entry, 0, true);
    *out_dir = iftGetStrValFromDict("--output-dir", args);
    if (!iftDirExists(*out_dir))
        iftMakeDir(*out_dir);
    
    puts("--------------------------");
    printf("- Test Asymmetry Map: %s\n", *asymmap_path);
    printf("- Fixed Mask: %s\n", *fix_mask_path);
    printf("- Symmetric Supervoxels: %s\n", *symm_super_path);
    printf("- Train Asymmetry Map: %s\n", train_asymmap_set_entry);
    printf("- Output Directory: %s\n", *out_dir);
    puts("--------------------------");
}


void iftGetOptionalArgs(  iftDict *args, int *n_bins) {
    if (iftDictContainKey("--num-of-bins", args, NULL))
        *n_bins = iftGetLongValFromDict("--num-of-bins", args);
    else *n_bins = 128;

    printf("- Number of Bins: %d\n", *n_bins);
    puts("--------------------------\n");
}


iftDataSet **iftExtractNativeSupervoxelHAAFeats(  iftImage *test_asymmap,   iftImage *fix_mask,   iftImage *symm_super_img,
                                                  iftFileSet *train_asymmap_set, int n_bins, int *n_supervoxels_out) {
    iftImage *fix_super_img = iftMask(symm_super_img, fix_mask);
    int n_supervoxels = iftMaximumValue(fix_super_img);
    int n_samples = train_asymmap_set->n + 1; // training samples + testing sample
    
    iftDataSet **Zarr = iftAlloc(n_supervoxels + 1, sizeof(iftDataSet));
    
    for (int label = 1; label <= n_supervoxels; label++)
        Zarr[label] = iftCreateDataSet(n_samples, n_bins);

    #pragma omp parallel for
    for (long s = 0; s < train_asymmap_set->n; s++) {
        iftImage *train_asym_map = iftReadImageByExt(train_asymmap_set->files[s]->path);
        iftSupervoxelHAAFeatsInAsymMap(train_asym_map, fix_super_img, Zarr, s, n_bins, n_supervoxels);
        iftDestroyImage(&train_asym_map);
    }
    
    // testing sample is the last sample (index [n_samples - 1]) in all supervoxel datasets
    iftSupervoxelHAAFeatsInAsymMap(test_asymmap, fix_super_img, Zarr, n_samples - 1, n_bins, n_supervoxels);
    
    iftDestroyImage(&fix_super_img);
    
    if (n_supervoxels_out)
        *n_supervoxels_out = n_supervoxels;
    
    return Zarr;
}
/*************************************************************/









