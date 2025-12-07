#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, char **asymmap_path, char **mask_path,
                        int *mov_obj, int *fix_obj, char **symm_super_path, char **symm_super_template_path,
                        iftFileSet **train_set, iftFileSet **train_asymmap_set, char **out_dir);
void iftGetOptionalArgs(  iftDict *args, int *n_bins, char **template_native_path);

iftDataSet **iftExtractNativeSupervoxelHAAFeats(  iftImage *test_asymmap,   iftImage *fix_mask,
                                                  iftImage *symm_super_img,   iftFileSet *train_asymmap_set,
                                                  iftImage *symm_super_template_img, int n_bins,
                                                int *n_supervoxels_out);
iftImage *iftFilterSymmSupervoxels(  iftImage *img,   iftImage *template_native,   iftImage *symm_super_img,
                                     iftImage *mov_mask,   iftImage *fix_mask);
void iftBuildRefDataNativeSupervoxelDataSets(iftDataSet **Zarr, int n_supervoxels, const char *test_img_path,
                                             const char *test_supervoxels_path, const char *test_supervoxels_template_path,
                                               iftFileSet *train_set);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);
    
    char *img_path = NULL;
    char *asymmap_path = NULL;
    char *template_native_path = NULL;
    char *mask_path = NULL;
    int mov_obj;
    int fix_obj;
    char *symm_super_path = NULL;
    char *symm_super_template_path = NULL;
    iftFileSet *train_set = NULL;
    iftFileSet *train_asymmap_set = NULL;
    char *out_dir = NULL;
    // optional args
    int n_bins;
    
    iftGetRequiredArgs(args, &img_path, &asymmap_path, &mask_path, &mov_obj, &fix_obj, &symm_super_path,
                       &symm_super_template_path, &train_set, &train_asymmap_set, &out_dir);
    iftGetOptionalArgs(args, &n_bins, &template_native_path);
    
    timer *t1 = iftTic();
    
    iftImage *img = iftReadImageByExt(img_path);
    iftImage *asymmap = iftReadImageByExt(asymmap_path);
    iftImage *mask = iftReadImageByExt(mask_path);
    iftImage *mov_mask = iftExtractObject(mask, mov_obj);
    iftImage *fix_mask = iftExtractObject(mask, fix_obj);
    iftImage *symm_super_img = iftReadImageByExt(symm_super_path);
    iftImage *symm_super_template_img = iftReadImageByExt(symm_super_template_path);
    
    puts("- Extracting HAA Feats");
    int n_supervoxels;
    iftDataSet **Zarr = iftExtractNativeSupervoxelHAAFeats(asymmap, fix_mask, symm_super_img, train_asymmap_set,
                                                           symm_super_template_img, n_bins, &n_supervoxels);
    
    char *img_id = iftFilename(img_path, iftFileExt(img_path));
    
    puts("- Building Reference Data");
    if (template_native_path && iftFileExists(template_native_path)) {
        iftImage *template_native = iftReadImageByExt(template_native_path);
    
        iftImage *super_img = iftFilterSymmSupervoxels(img, template_native, symm_super_img, mov_mask, fix_mask);
        char *super_img_path = iftJoinPathnames(3, out_dir, "FilteredSymmSupervoxels", iftCopyString("%s.nii.gz", img_id));
        iftWriteImageByExt(super_img, super_img_path);
        
        iftBuildRefDataSupervoxelDataSets(Zarr, n_supervoxels, img_path, super_img_path, train_set);
        
        iftDestroyImage(&template_native);
        iftDestroyImage(&super_img);
    }
    else iftBuildRefDataSupervoxelDataSets(Zarr, n_supervoxels, img_path, symm_super_path, train_set);
    
    puts("- Writing Supervoxel Datasets");
    char *out_datasets_dir = iftJoinPathnames(4, out_dir, iftCopyString("NumOfBins_%d", n_bins), "DataSets", img_id);
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
    iftDestroyImage(&img);
    iftDestroyImage(&asymmap);
    iftDestroyImage(&mask);
    iftDestroyImage(&mov_mask);
    iftDestroyImage(&fix_mask);
    iftDestroyImage(&symm_super_img);
    iftDestroyImage(&symm_super_template_img);
    
    
    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Extracts the histogram of absolute asymmetries (HAA) between each pair of symmetric supervoxels of a given test image.\n" \
        "- For each pair of symmetric supervoxels, it creates a dataset where such pair is a sample.\n" \
        "- Each dataset has one sample extracted for the test image (on its own space) and one sample per training image " \
        "(on a template space) from a training control set.\n" \
        "- The filename of each dataset is supervoxel_XXXX.zip, where XXXX is the supervoxel label, starting at 0001.";
    
    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-t", .long_name = "--test-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the Test Image."},
        {.short_name = "-a", .long_name = "--test-asymmetry-map", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname to save the test asymmetry map."},
        {.short_name = "-m", .long_name = "--mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname of the Mask."},
        {.short_name = "-l", .long_name = "--pairs-of-symmetric-objects", .has_arg=true, .arg_type=IFT_STR_TYPE,
        .required=true, .help="Define the a pair of labels for registration: The first is registered on to the second.\n" \
                              "e.g. 4:3 (object 4 is registered on to 3)."},
        {.short_name = "-s", .long_name = "--symm-super-path", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the symmetric supervoxel image."},
        {.short_name = "-v", .long_name = "--symm-super-on-template-path", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the symmetric supervoxel image on TEMPLATE space."},
        {.short_name = "-i", .long_name = "--train-set", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Dir. or CSV with the pathname from the Train. Images on TEMPLATE space."},
        {.short_name = "-j", .long_name = "--train-asym-maps", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Dir. or CSV with the pathname from the Asymmetry maps from the training images."},
        {.short_name = "-o", .long_name = "--output-dir", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Directory."},
        {.short_name = "-n", .long_name = "--num-of-bins", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=false, .help="Number of bins of the histogram. Default: 128."},
        {.short_name = "-r", .long_name = "--template-on-native", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Pathname of the Template registered on the NATIVE space."},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);
    
    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);
    
    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **img_path, char **asymmap_path, char **mask_path,
                        int *mov_obj, int *fix_obj, char **symm_super_path, char **symm_super_template_path,
                        iftFileSet **train_set, iftFileSet **train_asymmap_set, char **out_dir) {
    *img_path = iftGetStrValFromDict("--test-image", args);
    *asymmap_path = iftGetStrValFromDict("--test-asymmetry-map", args);
    *mask_path = iftGetStrValFromDict("--mask", args);
    *symm_super_path = iftGetStrValFromDict("--symm-super-path", args);
    *symm_super_template_path = iftGetStrValFromDict("--symm-super-on-template-path", args);
    const char *train_set_entry = iftGetConstStrValFromDict("--train-set", args);
    *train_set = iftLoadFileSetFromDirOrCSV(train_set_entry, 0, true);
    const char *train_asymmap_set_entry = iftGetConstStrValFromDict("--train-asym-maps", args);
    *train_asymmap_set = iftLoadFileSetFromDirOrCSV(train_asymmap_set_entry, 0, true);
    *out_dir = iftGetStrValFromDict("--output-dir", args);
    if (!iftDirExists(*out_dir))
        iftMakeDir(*out_dir);
    
    const char *symm_objs_entry = iftGetConstStrValFromDict("--pairs-of-symmetric-objects", args);
    
    if (iftRegexMatch(symm_objs_entry, "^[0-9]+:[0-9]+$")) {
        *mov_obj = atoi(iftSplitStringAt(symm_objs_entry, ":", 0));
        *fix_obj = atoi(iftSplitStringAt(symm_objs_entry, ":", 1));
        
        if (*fix_obj == *mov_obj)
            iftError("First Object label is equal to the second Object Label: %d",
                     "iftGetInputArgs", *fix_obj);
    } else
        iftError("Pair of Symmetric Objects \"%s\" does not match with the expected regular expression: [0-9]+:[0-9]+",
                 "iftGetInputArgs", symm_objs_entry);
    
    puts("--------------------------");
    printf("- Test Image: %s\n", *img_path);
    printf("- Test Asymmetry Map: %s\n", *asymmap_path);
    printf("- Mask: %s\n", *mask_path);
    printf("- Moving Object: %d\n", *mov_obj);
    printf("- Fixed Object: %d\n", *fix_obj);
    printf("- Symmetric Supervoxels on NATIVE: %s\n", *symm_super_path);
    printf("- Symmetric Supervoxels on TEMPLATE: %s\n", *symm_super_template_path);
    printf("- Train Set: %s\n", train_set_entry);
    printf("- Train Asymmetry Map: %s\n", train_asymmap_set_entry);
    printf("- Output Directory: %s\n", *out_dir);
    puts("--------------------------");
}


void iftGetOptionalArgs(  iftDict *args, int *n_bins, char **template_native_path) {
    if (iftDictContainKey("--num-of-bins", args, NULL))
        *n_bins = iftGetLongValFromDict("--num-of-bins", args);
    else *n_bins = 128;
    
    *template_native_path = iftGetStrValFromDict("--template-on-native", args);
    
    printf("- Number of Bins: %d\n", *n_bins);
    printf("- Template on NATIVE space: %s\n", *template_native_path);
    puts("--------------------------\n");
}


iftDataSet **iftExtractNativeSupervoxelHAAFeats(  iftImage *test_asymmap,   iftImage *fix_mask,
                                                  iftImage *symm_super_img,   iftFileSet *train_asymmap_set,
                                                  iftImage *symm_super_template_img, int n_bins,
                                                int *n_supervoxels_out) {
    iftImage *fix_super_img = iftMask(symm_super_img, fix_mask);
    iftImage *fix_super_template_img = iftExtractBrainSide(symm_super_template_img, IFT_RIGHT_BRAIN_SIDE);
    
    int n_supervoxels = iftMaximumValue(fix_super_img);
    int n_samples = train_asymmap_set->n + 1; // training samples + testing sample
    
    iftDataSet **Zarr = iftAlloc(n_supervoxels + 1, sizeof(iftDataSet));
    
    for (int label = 1; label <= n_supervoxels; label++)
        Zarr[label] = iftCreateDataSet(n_samples, n_bins);

    #pragma omp parallel for
    for (long s = 0; s < train_asymmap_set->n; s++) {
        iftImage *train_asym_map = iftReadImageByExt(train_asymmap_set->files[s]->path);
        iftSupervoxelHAAFeatsInAsymMap(train_asym_map, fix_super_template_img, Zarr, s, n_bins, n_supervoxels);
        iftDestroyImage(&train_asym_map);
    }
    
    // testing sample is the last sample (index [n_samples - 1]) in all supervoxel datasets
    iftSupervoxelHAAFeatsInAsymMap(test_asymmap, fix_super_img, Zarr, n_samples - 1, n_bins, n_supervoxels);
    
    iftDestroyImage(&fix_super_img);
    iftDestroyImage(&fix_super_template_img);
    
    if (n_supervoxels_out)
        *n_supervoxels_out = n_supervoxels;
    
    return Zarr;
}


iftImage *iftFilterSymmSupervoxels(  iftImage *img,   iftImage *template_native,   iftImage *symm_super_img,
                                     iftImage *mov_mask,   iftImage *fix_mask) {
    iftImage *asymmap_ref = iftAbsSub(img, template_native);
    iftImage *super_mov_img = iftMask(symm_super_img, mov_mask);
    iftImage *super_fix_img = iftMask(symm_super_img, fix_mask);
    
    iftFloatArray *means_asyms_mov = iftMeanValueInRegions(asymmap_ref, super_mov_img);
    iftFloatArray *means_asyms_fix = iftMeanValueInRegions(asymmap_ref, super_fix_img);
    long n_supervoxels = means_asyms_mov->n - 1;
    
    iftIntArray *switch_mov = iftCreateIntArray(means_asyms_mov->n);
    iftIntArray *switch_fix = iftCreateIntArray(means_asyms_fix->n);

    #pragma omp parallel for
    for (long label = 1; label <= n_supervoxels; label++) {
        if (means_asyms_mov->val[label] >= means_asyms_fix->val[label]) {
            switch_mov->val[label] = 1;
            switch_fix->val[label] = 0;
        }
        else {
            switch_mov->val[label] = 0;
            switch_fix->val[label] = 1;
        }
    }
    
    iftImage *super_img_filt = iftCreateImageFromImage(symm_super_img);
    
    // since there is no intersection between super_mov_img and super_fix_img, we can sum them after turning on/off
    // their values at each voxel to get the final supervoxel image
    #pragma omp parallel for
    for (int p = 0; p < symm_super_img->n; p++) {
        super_mov_img->val[p] *= switch_mov->val[super_mov_img->val[p]];
        super_fix_img->val[p] *= switch_fix->val[super_fix_img->val[p]];
        super_img_filt->val[p] = super_mov_img->val[p] + super_fix_img->val[p];
    }
    
    iftDestroyImage(&asymmap_ref);
    iftDestroyImage(&super_mov_img);
    iftDestroyImage(&super_fix_img);
    iftDestroyFloatArray(&means_asyms_mov);
    iftDestroyFloatArray(&means_asyms_fix);
    iftDestroyIntArray(&switch_mov);
    iftDestroyIntArray(&switch_fix);
    
    return super_img_filt;
}


void iftBuildRefDataNativeSupervoxelDataSets(iftDataSet **Zarr, int n_supervoxels, const char *test_img_path,
                                             const char *test_supervoxels_path, const char *test_supervoxels_template_path,
                                               iftFileSet *train_set) {
    #pragma omp parallel for
    for (int label = 1; label <= n_supervoxels; label++) {
        iftCSV *ref_data = iftCreateCSV(Zarr[label]->nsamples, 3);
        iftSetCSVHeader(ref_data, "image_path,symmetric_supervoxel_path,supervoxel_label");
        
        if (Zarr[label]->nsamples != (train_set->n + 1))
            iftError("Number of samples for Supervoxel DataSet [%d] is != from the number of training samples + 1 ==> %d ! %d",
                     "iftBuildRefDataSupervoxelDataSets", label, Zarr[label]->nsamples, train_set->n + 1);

        #pragma omp parallel for
        for (int s = 0; s < train_set->n; s++) {
            strcpy(ref_data->data[s][0], train_set->files[s]->path);
            strcpy(ref_data->data[s][1], test_supervoxels_template_path);
            sprintf(ref_data->data[s][2], "%d", label);
        }
        
        strcpy(ref_data->data[train_set->n][0], test_img_path);
        strcpy(ref_data->data[train_set->n][1], test_supervoxels_path);
        sprintf(ref_data->data[train_set->n][2], "%d", label);
        
        iftSetRefData(Zarr[label], ref_data, IFT_REF_DATA_CSV);
        
        iftDestroyCSV(&ref_data);
    }
}
/*************************************************************/









