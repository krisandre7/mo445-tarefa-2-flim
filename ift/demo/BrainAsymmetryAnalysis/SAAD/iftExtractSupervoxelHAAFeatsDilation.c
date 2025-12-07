//
// Created by Samuel Martins on 11/12/18.
//

#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **test_img_path, char **test_super_path, iftFileSet **train_set,
                        char **out_dir);
void iftGetOptionalArgs(  iftDict *args, float *dilation_radius, int *n_bins,
                        char **normal_asym_map_path);

void iftSingleSupervoxelHAAFeatsInAsymMap(  iftImage *asym_map,   iftImage *bin_mask,
                                          iftDataSet *Z, int s, int n_bins, int n_supervoxels);
iftDataSet **iftExtractSupervoxelHAAFeatsDilation(  iftImage *test_img,   iftImage *test_sym_svoxels,
                                                    iftFileSet *train_set, int n_bins,   iftImage *normal_asym_map,
                                                  float radius, int *n_svoxels_out);
/*************************************************************/


int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    char *test_img_path = NULL;
    char *test_super_path = NULL;
    iftFileSet *train_set = NULL;
    char *out_dir = NULL;
    // optional args
    float dilation_radius;
    int n_bins;
    char *normal_asym_map_path = NULL;
    
    iftGetRequiredArgs(args, &test_img_path, &test_super_path, &train_set, &out_dir);
    iftGetOptionalArgs(args, &dilation_radius, &n_bins, &normal_asym_map_path);
    
    timer *t1 = iftTic();

    iftImage *test_img = iftReadImageByExt(test_img_path);
    iftImage *test_supervoxels = iftReadImageByExt(test_super_path);
    iftImage *normal_asym_map = (normal_asym_map_path) ? iftReadImageByExt(normal_asym_map_path) : NULL;
    
    puts("- Building Datasets");
    int n_supervoxels;
    iftDataSet **Zarr = iftExtractSupervoxelHAAFeatsDilation(test_img, test_supervoxels, train_set, n_bins, normal_asym_map, dilation_radius, &n_supervoxels);
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
    iftDestroyImage(&normal_asym_map);

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
        {.short_name = "-i", .long_name = "--test-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Test Image used to build the supervoxel datasets."},
        {.short_name = "-s", .long_name = "--test-symmetric-supervoxels", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Label Image with the symmetric supervoxels."},
        {.short_name = "-t", .long_name = "--training-set", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Directory or CSV with the pathnames from the training control images."},
        {.short_name = "-o", .long_name = "--output-dir", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Directory where the symmetry supervoxel datasets will be saved."},
        {.short_name = "-r", .long_name = "--dilation-radius", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Dilation Radius. Default: 1.0."}, 
        {.short_name = "-n", .long_name = "--num-of-bins", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=false, .help="Number of bins of the histogram. Default: 128."},
        {.short_name = "-m", .long_name = "--normal-asymmetry-map", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Normal asymmetry map used to attenuate the testing image asymmetries."},
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


void iftGetOptionalArgs(  iftDict *args, float *dilation_radius, int *n_bins,
                        char **normal_asym_map_path) {
    if (iftDictContainKey("--dilation-radius", args, NULL))
        *dilation_radius = iftGetDblValFromDict("--dilation-radius", args);
    else
        *dilation_radius = 1.0;
    printf("- Dilation Radius: %f\n", *dilation_radius);

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

void iftSingleSupervoxelHAAFeatsInAsymMap(  iftImage *asym_map,   iftImage *bin_mask,
                                          iftDataSet *Z, int s, int n_bins, int n_supervoxels)
{
    iftHist *hist = iftCalcGrayImageHist(asym_map, bin_mask, n_bins, true);

    if (Z->nfeats != n_bins)
        iftError("Number of Feats for Supervoxel DataSet is != from the number of histogram bins: %d != %d",
                 "iftSupervoxelHAAFeatsInAsymMap", Z->nfeats, n_bins);

#pragma omp parallel for
    for (int f = 0; f < Z->nfeats; f++)
    {
        Z->sample[s].feat[f] = hist->val[f];
    }

    iftDestroyHist(&hist);
}

iftDataSet **iftExtractSupervoxelHAAFeatsDilation(  iftImage *test_img,   iftImage *test_sym_svoxels,
                                                    iftFileSet *train_set, int n_bins,   iftImage *normal_asym_map,
                                                  float radius, int *n_svoxels_out)
{
    iftImage *svoxels_right_side = iftExtractBrainSide(test_sym_svoxels, IFT_RIGHT_BRAIN_SIDE);

    int n_svoxels = iftMaximumValue(svoxels_right_side);
    int n_samples = train_set->n + 1; // training samples + testing sample

    iftDataSet **Zarr = iftAlloc(n_svoxels + 1, sizeof(iftDataSet *));
    iftIntArray *labels = iftGetObjectLabels(test_sym_svoxels);
    iftImageArray *dilate_imgs = iftCreateImageArray(n_svoxels + 1);

#pragma omp parallel for
    for (int o = 0; o < labels->n; o++)
    {
        int label = labels->val[o];
        Zarr[label] = iftCreateDataSet(n_samples, n_bins);
        iftImage *obj_mask = iftExtractObject(svoxels_right_side, label);

        iftSet *S = NULL;
        dilate_imgs->val[label] = iftDilateBin(obj_mask, &S, radius);
        // iftWriteImageByExt(dilate_imgs->val[label], "tmp/dilate_%d.nii.gz", label);

        iftDestroySet(&S);
        iftDestroyImage(&obj_mask);
    }

    for (int label = 1; label <= n_svoxels; label++)
        Zarr[label] = iftCreateDataSet(n_samples, n_bins);

#pragma omp parallel for
    for (long s = 0; s < train_set->n; s++)
    {
        iftImage *train_img = iftReadImageByExt(train_set->files[s]->path);
        iftImage *train_asym_map = iftBrainAsymMap(train_img, NULL);

        for (int o = 0; o < labels->n; o++)
        {
            int label = labels->val[o];
            if (dilate_imgs->val[label] == NULL)
                iftError("Dilated Image is NULL", "iftExtractSupervoxelHistRegErrorsFeatsDilation");
            if (Zarr[label] == NULL)
                iftError("Dataset is NULL", "iftExtractSupervoxelHistRegErrorsFeatsDilation");

            iftSingleSupervoxelHAAFeatsInAsymMap(train_asym_map, dilate_imgs->val[label],
                                                 Zarr[label], s, n_bins, n_svoxels);
        }

        iftDestroyImage(&train_img);
        iftDestroyImage(&train_asym_map);
    }

    // testing sample is the last sample (index [n_samples - 1]) in all supervoxel datasets
    iftImage *test_asym_map = iftBrainAsymMap(test_img, normal_asym_map);
#pragma omp parallel for
    for (int o = 0; o < labels->n; o++)
    {
        int label = labels->val[o];
        iftSingleSupervoxelHAAFeatsInAsymMap(test_asym_map, dilate_imgs->val[label],
                                             Zarr[label], n_samples - 1, n_bins, n_svoxels);
    }
    iftDestroyImage(&test_asym_map);

    iftDestroyImage(&svoxels_right_side);
    iftDestroyIntArray(&labels);
    iftDestroyImageArray(&dilate_imgs);

    if (n_svoxels_out)
        *n_svoxels_out = n_svoxels;

    return Zarr;
}
/*************************************************************/




