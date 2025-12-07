//
// Created by Samuel Martins on 11/12/18.
//

#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **test_img_path, char **test_super_path, iftFileSet **train_set,
                        char **out_dir);
void iftGetOptionalArgs(  iftDict *args, int *n_bins);

void iftSupervoxelHRHLFeatsInImage(  iftImage *img,   iftImage *svoxels_right_side,
                                     iftImage *svoxels_left_side, iftDataSet **Zarr, int s, int n_bins,
                                   int max_val, int n_supervoxels);
iftDataSet **iftExtractSupervoxelHRHLFeats(  iftImage *test_img,   iftImage *test_sym_svoxels,
                                             iftFileSet *train_set, int n_bins, int *n_svoxels_out);
/*************************************************************/


int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    char *test_img_path = NULL;
    char *test_super_path = NULL;
    iftFileSet *train_set = NULL;
    char *out_dir = NULL;
    // optional args
    int n_bins;
    
    iftGetRequiredArgs(args, &test_img_path, &test_super_path, &train_set, &out_dir);
    iftGetOptionalArgs(args, &n_bins);
    
    timer *t1 = iftTic();

    iftImage *test_img = iftReadImageByExt(test_img_path);
    iftImage *test_supervoxels = iftReadImageByExt(test_super_path);
    
    puts("- Building Datasets");
    int n_supervoxels;
    iftDataSet **Zarr = iftExtractSupervoxelHRHLFeats(test_img, test_supervoxels, train_set, n_bins, &n_supervoxels);
    puts("- Building Reference Data");
    iftBuildRefDataSupervoxelDataSets(Zarr, n_supervoxels, test_img_path, test_super_path, train_set);
    
    puts("- Writing Supervoxel Datasets");
    #pragma omp parallel for
    for (int label = 1; label <= n_supervoxels; label++) {
        char *filename = iftCopyString("supervoxel_%04d.zip", label);
        char *out_dataset_path = iftJoinPathnames(2, out_dir, filename);
        
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
    iftDestroyFileSet(&train_set);
    iftDestroyImage(&test_img);
    iftDestroyImage(&test_supervoxels);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Extracts the histogram of left and right hemispheres from each pair of symmetric supervoxels of a given test image.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--test-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Test Image used to build the supervoxel datasets."},
        {.short_name = "-s", .long_name = "--test-symmetric-supervoxels", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Label Image with the symmetric supervoxels."},
        {.short_name = "-t", .long_name = "--training-set", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Directory or CSV with the pathnames from the training control images."},
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


void iftGetRequiredArgs(  iftDict *args, char **test_img_path, char **test_super_path, iftFileSet **train_set,
                        char **out_dir) {
    *test_img_path = iftGetStrValFromDict("--test-image", args);
    *test_super_path = iftGetStrValFromDict("--test-symmetric-supervoxels", args);
    const char *train_entry = iftGetConstStrValFromDict("--training-set", args);
    *train_set = iftLoadFileSetFromDirOrCSV(train_entry, 0, true);
    *out_dir = iftGetStrValFromDict("--output-dir", args);
    
    if (!iftDirExists(*out_dir))
        iftMakeDir(*out_dir);
    
    puts("--------------------------");
    printf("- Test Image: %s\n", *test_img_path);
    printf("- Test Symmetric Supervoxels: %s\n", *test_super_path);
    printf("- Train Images Entry: %s\n", train_entry);
    printf("- Output Dir: %s\n", *out_dir);
    puts("--------------------------");
}


void iftGetOptionalArgs(  iftDict *args, int *n_bins) {
    if (iftDictContainKey("--num-of-bins", args, NULL))
        *n_bins = (int) iftGetLongValFromDict("--num-of-bins", args);
    else *n_bins = 128;
    printf("- Num of Bins: %d\n", *n_bins);
    puts("--------------------------\n");
}



void iftSupervoxelHRHLFeatsInImage(  iftImage *img,   iftImage *svoxels_right_side,
                                     iftImage *svoxels_left_side, iftDataSet **Zarr, int s, int n_bins,
                                   int max_val, int n_supervoxels) {
    iftHist **hists_right = iftCalcGrayImageHistForLabels(img, svoxels_right_side, n_bins, max_val, true, NULL); // size n_supervoxels + 1
    iftHist **hists_left = iftCalcGrayImageHistForLabels(img, svoxels_left_side, n_bins, max_val, true, NULL);   // size n_supervoxels + 1

    #pragma omp parallel for
    for (int label = 1; label <= n_supervoxels; label++) {
        if (Zarr[label]->nfeats != (2 * n_bins)) {
            iftError("Number of Feats for Supervoxel DataSet [%d] is != from the 2 * number of histogram bins: %d != %d",
                        "iftSupervoxelHRHLFeatsInImage", label, Zarr[label]->nfeats, 2 * n_bins);
        }

        for (int b = 0; b < n_bins; b++) {
            Zarr[label]->sample[s].feat[b] = hists_right[label]->val[b];
            Zarr[label]->sample[s].feat[b + n_bins] = hists_left[label]->val[b];

        }
        iftDestroyHist(&hists_right[label]);
        iftDestroyHist(&hists_left[label]);
    }
    iftDestroyHist(&hists_right[0]);
    iftDestroyHist(&hists_left[0]);
    iftFree(hists_right);
    iftFree(hists_left);
}


iftDataSet **iftExtractSupervoxelHRHLFeats(  iftImage *test_img,   iftImage *test_sym_svoxels,
                                             iftFileSet *train_set, int n_bins, int *n_svoxels_out) {
    iftImage *svoxels_right_side = iftExtractBrainSide(test_sym_svoxels, IFT_RIGHT_BRAIN_SIDE);
    iftImage *svoxels_left_side = iftExtractBrainSide(test_sym_svoxels, IFT_LEFT_BRAIN_SIDE);
    iftWriteImageByExt(svoxels_right_side, "tmp/svoxels_right_side.nii.gz");
    iftWriteImageByExt(svoxels_left_side, "tmp/svoxels_left_side.nii.gz");

    int norm_val = iftMaximumValue(test_img);

    int n_svoxels = iftMaximumValue(svoxels_right_side);
    int n_samples = train_set->n + 1; // training samples + testing sample

    iftDataSet **Zarr = iftAlloc(n_svoxels + 1, sizeof(iftDataSet *));

    for (int label = 1; label <= n_svoxels; label++)
        Zarr[label] = iftCreateDataSet(n_samples, 2 * n_bins);

    #pragma omp parallel for
    for (long s = 0; s < train_set->n; s++) {
        iftImage *train_img = iftReadImageByExt(train_set->files[s]->path);
        iftSupervoxelHRHLFeatsInImage(train_img, svoxels_right_side, svoxels_left_side, Zarr, s, n_bins, norm_val, n_svoxels);
        iftDestroyImage(&train_img);
    }

    // testing sample is the last sample (index [n_samples - 1]) in all supervoxel datasets
    iftSupervoxelHRHLFeatsInImage(test_img, svoxels_right_side, svoxels_left_side,  Zarr, n_samples - 1, n_bins, norm_val, n_svoxels);

    iftDestroyImage(&svoxels_right_side);
    iftDestroyImage(&svoxels_left_side);

    if (n_svoxels_out) {
        *n_svoxels_out = n_svoxels;
    }

    return Zarr;
}
/*************************************************************/




