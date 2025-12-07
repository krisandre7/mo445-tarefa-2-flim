#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **test_img_path, char **test_super_path, iftFileSet **train_set,
                        char **out_dir);
void iftGetOptionalArgs(  iftDict *args, int *n_bins_per_channel, char **normal_asym_map_path);
/*************************************************************/


int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    char *test_img_path = NULL;
    char *test_super_path = NULL;
    iftFileSet *train_set = NULL;
    char *out_dir = NULL;
    // optional args
    int n_bins_per_channel;
    char *normal_asym_map_path = NULL;
    
    iftGetRequiredArgs(args, &test_img_path, &test_super_path, &train_set, &out_dir);
    iftGetOptionalArgs(args, &n_bins_per_channel, &normal_asym_map_path);

    timer *t1 = iftTic();

    iftImage *test_img = iftReadImageByExt(test_img_path);
    iftImage *test_supervoxels = iftReadImageByExt(test_super_path);
    iftImage *normal_asym_map = (normal_asym_map_path) ? iftReadImageByExt(normal_asym_map_path) : NULL;
    
    puts("- Building Datasets");
    int n_supervoxels;
    iftDataSet **Zarr = iftExtractSupervoxelBICAsymmFeats(test_img, test_supervoxels, train_set, n_bins_per_channel, normal_asym_map, &n_supervoxels);
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
        "- Extracts the BIC between each pair of symmetric supervoxels of a given test image.\n" \
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
        {.short_name = "-n", .long_name = "--num-bins-per-channel", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=false, .help=" Number of bins per image band/channel for BIC. Default: 8."},
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


void iftGetOptionalArgs(  iftDict *args, int *n_bins_per_channel, char **normal_asym_map_path) {
    if (iftDictContainKey("--num-bins-per-channel", args, NULL))
        *n_bins_per_channel = (int) iftGetLongValFromDict("--num-bins-per-channel", args);
    else *n_bins_per_channel = 8;
    printf("- Number of bins per image band/channel for BIC: %d\n", *n_bins_per_channel);

    if (iftDictContainKey("--normal-asymmetry-map", args, NULL)) {
        *normal_asym_map_path = iftGetStrValFromDict("--normal-asymmetry-map", args);
        printf("- Normal Asymmetry Map: %s\n", *normal_asym_map_path);
    }
    else *normal_asym_map_path = NULL;
    
    puts("--------------------------\n");
}
/*************************************************************/




