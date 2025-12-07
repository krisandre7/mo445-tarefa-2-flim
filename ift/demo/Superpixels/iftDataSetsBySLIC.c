/**
 * @file
 * @brief Generates Supervoxels by SLIC for the Input Images and builds two datasets with the supervoxels:
 * 1) multi-label dataset with voxel coords from each supervoxel's centroids/roots as feature vectors.
 * 2) binary dataset (object or background) with brightness information (only voxel's value) from each supervoxel as feature vectors.
 * PS: It is expected that all images are registered.
 * PS2: The true label of each supervoxel is the one with majority voting + 1 to guarantee that
 * the background (true label 0) will be 1.
 * @note See the source code in @ref iftDataSetsBySlic.c
 *
 * @example iftDataSetsBySlic.c
 * @brief Generates Supervoxels by SLIC for the Input Images and builds two datasets with the supervoxels:
 * 1) multi-label dataset with voxel coords from each supervoxel's centroids/roots as feature vectors.
 * 2) binary dataset (object or background) with brightness information (only voxel's value)  from each supervoxel as feature vectors.
 * PS: It is expected that all images are registered.
 * PS2: The true label of each supervoxel is the one with majority voting + 1 to guarantee that
 * the background (true label 0) will be 1.
 * @author Samuel Martins
 * @date Jun 17, 2016
 */


#include "ift.h"

/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, iftFileSet **img_set, iftDict **label_dict,
                        int *input_n_clusters, double *comp, char **out_base);
void iftValidateInputs(const char *img_entry, const char *label_entry, int input_n_clusters,
                       double comp, const char *out_base);
void iftValidateFileSet(  iftFileSet *img_files, const char *img_entry);
void iftGetOptionalArgs(  iftDict *args, const char *img_path, int *img_max_range,
                        double *sampling_perc, iftImageDomain *domain);
void iftBuildGlobalDataSet(iftDataSet **Zc_imgs, iftDataSet **Zt_imgs, size_t n_imgs, float sampling_perc,
                           bool balance_dataset, iftDataSet **Zc, iftDataSet **Zt);
void iftWriteDataSetAsImage(  iftDataSet *Zc,   iftDataSet *Zt, iftImageDomain domain, const char *out_base);
/*************************************************************/



int main(int argc, const char *argv[]) {
    iftRandomSeed(time(NULL));

    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    iftFileSet *img_set = NULL;
    iftDict *label_dict = NULL; // dict with all label pathnames indexed by their file prefixes
    int input_n_clusters;
    double comp;
    char *out_base;
    // optional args
    int img_max_range;
    double sampling_perc;
    iftImageDomain domain;

    iftGetRequiredArgs(args, &img_set, &label_dict, &input_n_clusters, &comp, &out_base);
    iftGetOptionalArgs(args, img_set->files[0]->path, &img_max_range, &sampling_perc, &domain);

    timer *t1 = iftTic();

      size_t n_imgs  = img_set->n;
    // prob. map of a voxel is an object (any one of them)
    iftFImage *prob_map  = iftCreateFImage(domain.xsize, domain.ysize, domain.zsize);
    iftDataSet **Zc_imgs = (iftDataSet**) iftAlloc(n_imgs, sizeof(iftDataSet*));
    iftDataSet **Zt_imgs = (iftDataSet**) iftAlloc(n_imgs, sizeof(iftDataSet*));

    // for each input image
    #pragma omp parallel for schedule(auto)
    for (size_t i = 0; i < n_imgs; i++) {
        const char *img_path   = img_set->files[i]->path;
        char *img_key          = iftFileKey(img_path);
        const char *label_path = iftGetConstStrValFromDict(img_key, label_dict);
        iftFree(img_key);

        printf("- Thread: %d: [%lu/%lu]\n%s\n%s\n\n", omp_get_thread_num(), i+1, n_imgs, img_path, label_path);

        int n_clusters; // final number of clusters/supervoxels
        iftImage *img       = iftReadImageByExt(img_path);
        iftImage *label_img = iftReadImageByExt(label_path);
        iftImage *super_img = iftGenerateSuperpixelsBySlic(img, NULL, input_n_clusters, comp, img_max_range, &n_clusters);
        printf("- Thread: %d: [%lu/%lu] final_n_supervoxels: %d\n", omp_get_thread_num(), i+1, n_imgs, n_clusters);

        iftIntArray *true_labels = iftGetSupervoxelTrueLabelByMajorityVoting(label_img, super_img);
        iftIntArray *bin_labels  = iftCreateIntArray(true_labels->n);
        // char out_path[512];
        // sprintf(out_path, "super/%04lu.zip", i);
        // iftWriteSuperpixelBorders(img, super_img, out_path);

        // - binary labels: 1 for background, 2 for any object
        // - sums 1 to true_labels to guarantee that the background (label 0) is 1
        for (int c = 0; c < true_labels->n; c++) {
            bin_labels->val[c] = (true_labels->val[c] != 0) + 1;
            true_labels->val[c]++;
        }

        iftVoxel *roots = iftGeodesicCenterVoxelCoords(super_img, NULL);

        Zc_imgs[i] = iftSupervoxelsToVoxelCoordsDataSet(super_img, false, roots, true_labels);
        Zt_imgs[i] = iftSupervoxelsToDataSet(img, super_img, bin_labels);
//        Zt_imgs[i] = iftSupervoxelsToGrayAvgStdevMinMaxDataSet(img, super_img, img_max_range, false, bin_labels);
        // Zt_imgs[i] = iftSupervoxelsToHistogramDataSet(img, super_img, 256, bin_labels);

        // counts the object voxels (bin_label 2)
        for (int p = 0; p < super_img->n; p++) {
            int c = super_img->val[p] - 1;
            #pragma omp atomic
            prob_map->val[p] += (true_labels->val[c] > 1);
        }

        iftDestroyImage(&img);
        iftDestroyImage(&label_img);
        iftDestroyImage(&super_img);
        iftDestroyIntArray(&true_labels);
        iftDestroyIntArray(&bin_labels);
        iftFree(roots);
    }


    puts("\n- Building the Global DataSet");
    iftDataSet *Zc = NULL;
    iftDataSet *Zt = NULL;
    iftBuildGlobalDataSet(Zc_imgs, Zt_imgs, n_imgs, sampling_perc, iftDictContainKey("--balance-datasets", args, NULL), &Zc, &Zt);

    puts("- Saving Coord. Dataset");
    char *out_path = iftConcatStrings(2, out_base, "_coords.zip");
    iftWriteOPFDataSet(Zc, out_path);
    iftFree(out_path);

    puts("- Saving Brightness (Texture) Dataset");
    out_path = iftConcatStrings(2, out_base, "_bright.zip");
    iftWriteOPFDataSet(Zt, out_path);
    iftFree(out_path);

    puts("- Saving Coordinate Image");
    iftWriteDataSetAsImage(Zc, Zt, domain, out_base);

    puts("- Saving Fuzzy Image");
    out_path               = iftConcatStrings(2, out_base, "_prob_map.scn");
    iftImage *prob_map_img = iftFImageToImage(prob_map, img_max_range);
    iftWriteImage(prob_map_img, out_path);
    iftFree(out_path);
    iftDestroyFImage(&prob_map);
    iftDestroyImage(&prob_map_img);


    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyFileSet(&img_set);
    iftDestroyDict(&label_dict);
    for (size_t i = 0; i < n_imgs; i++) {
        iftDestroyDataSet(&Zc_imgs[i]);
        iftDestroyDataSet(&Zt_imgs[i]);
    }
    iftFree(Zc_imgs);
    iftFree(Zt_imgs);
    iftDestroyDataSetWithNoFeatVectors(&Zc);
    iftDestroyDataSetWithNoFeatVectors(&Zt);

    return 0;
}



/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "Generates Supervoxels by SLIC for the Input Images and builds two datasets with the supervoxels:\n" \
        "1) multi-label dataset with voxel coords from each supervoxel's centroids/roots as feature vectors.\n" \
        "2) binary dataset (object or background) with brightness information (only voxel's value) from each supervoxel as feature vectors.\n\n" \
        "* PS: It is expected that all images are registered.\n" \
        "* PS2: The true label of each supervoxel is the one with majority voting + 1 to guarantee that " \
        "the background (true label 0) will be 1.\n" \
        "* PS3: The (Image, Label Image) MUST HAVE THE SAME Image Primary Key: \n" \
        "-----------------------\n" \
        "Ex:\n" \
        " - IFT\'s convention\n" \
        "(./imgs/000010_000001_Orig.scn, ./labels/000010_000001_Labels.scn)\n--> Primary Key: 000010_000001\n\n" \
        "- No convention (gets the basename)\n" \
        "(./imgs/batman_and_robin.scn, ./labels/batman_and_robin.scn)\n--> Primary Key: batman_and_robin\n" \
        "-----------------------";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--img-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Dir with the Input Images or a CSV file with their pathnames."},
        {.short_name = "-l", .long_name = "--label-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Dir with the Label Images or a CSV file with their pathnames."},
        {.short_name = "-k", .long_name = "--num-clusters", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=true, .help="Approximate Number of Superpixels per block."},
        {.short_name = "-c", .long_name = "--compactness", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=true, .help="Compactness: Relative Importance of color similarity and spatial proximity.\n" \
                                "Try values on a log scale (0.01, 0.1, 1, 10, 100) before refining " \
                                "around a chosen value."},
        {.short_name = "-o", .long_name = "--out-dataset-basename", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output basenames from the Data Sets."},
        {.short_name = "-j", .long_name = "--img-depth", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=false, .help="Input Image Depth in bits (8, 12, 16, ...)\n" \
                                "Default: it tries to find the image depth automatically from the First Image"},
        {.short_name = "-p", .long_name = "--sampling-percentage", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Sampling Percentage used to get select a set of supervoxels to build the resulting Data Set.\n" \
                                "Default: 0.50"},
        {.short_name = "", .long_name = "--balance-datasets", .has_arg=false,
         .required=false, .help="Applies a Balance Class Sampling in the resulting dataset."},
        {.short_name = "", .long_name = "--num-threads", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=false, .help="Number of Threads of the program.\nDefault: use all threads possible."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, iftFileSet **img_set, iftDict **label_dict,
                        int *input_n_clusters, double *comp, char **out_base) {
    const char *img_entry   = iftGetConstStrValFromDict("--img-entry", args);
    const char *label_entry = iftGetConstStrValFromDict("--label-entry", args);
    *input_n_clusters       = iftGetLongValFromDict("--num-clusters", args);
    *comp                   = iftGetDblValFromDict("--compactness", args);
    *out_base               = iftGetStrValFromDict("--out-dataset-basename", args);

    iftValidateInputs(img_entry, label_entry, *input_n_clusters, *comp, *out_base);

    *img_set              = iftLoadFileSetFromDirOrCSV(img_entry, 0, true);
    iftFileSet *label_set = iftLoadFileSetFromDirOrCSV(label_entry, 0, true);

    iftValidateFileSet(*img_set, img_entry);
    iftValidateFileSet(label_set, label_entry);

    *label_dict = iftFileSetToDict(label_set);
    iftDestroyFileSet(&label_set);


    puts("-----------------------");
    printf("- Image Entry: \"%s\"\n", img_entry);
    printf("- Label Image Entry: \"%s\"\n", label_entry);
    printf("- Input Desired Number of Superpixels/Supervoxels: %d\n", *input_n_clusters);
    printf("- Compactness: %lf\n", *comp);
    printf("- Output Basenames for the Data Sets: \"%s\"\n", *out_base);
    puts("-----------------------");
}


void iftValidateInputs(const char *img_entry, const char *label_entry, int input_n_clusters,
                       double comp, const char *out_base) {
    // image entry
    if (iftFileExists(img_entry)) {
        if (!iftEndsWith(img_entry, ".csv")) {
            iftError("The image entry file \"%s\" is not a CSV file\nTry *.csv", "iftValidateInputs", img_entry);
        }
    }
    else if (!iftDirExists(img_entry))
        iftError("Invalid Pathname for the Image Entry: \"%s\"\nTry a valid Directory or a CSV file",
                 "iftValidateInputs", img_entry);

    // label entry
    if (iftFileExists(label_entry)) {
        if (!iftEndsWith(label_entry, ".csv")) {
            iftError("The label image entry file \"%s\" is not a CSV file\nTry *.csv", "iftValidateInputs", label_entry);
        }
    }
    else if (!iftDirExists(label_entry))
        iftError("Invalid Pathname for the Label Image Entry: \"%s\"\nTry a valid Directory or a CSV file",
                 "iftValidateInputs", label_entry);

    if (input_n_clusters <= 0)
        iftError("Invalid Desired Number of Superpixels: %ld... Try > 0", "iftValidateInputs", input_n_clusters);

    if (comp <= 0.0)
        iftError("Invalid Compactness: %lf... Try > 0 (0.01, 0.1, 1, 10)", "iftValidateInputs", comp);

    if (iftDirExists(out_base))
        iftError("Out basename \"%s\" cannot be only a directory", "iftValidateInputs", out_base);

    char *parent_dir = iftParentDir(out_base);
    if (!iftDirExists(parent_dir)) {
        iftMakeDir(parent_dir);
    }
    iftFree(parent_dir);
}


void iftValidateFileSet(  iftFileSet *img_files, const char *img_entry) {
    if (img_files->n == 0)
        iftError("There are no Images in \"%s\"", "iftValidateFileSet", img_entry);
    else {
        #pragma omp parallel for
        for (size_t i = 0; i < img_files->n; i++)
            if (!iftIsImageFile(img_files->files[i]->path))
                iftError("File \"%s\" is not an Image", "iftValidateFileSet", img_files->files[i]->path);
    }
}


void iftGetOptionalArgs(  iftDict *args, const char *img_path, int *img_max_range,
                        double *sampling_perc, iftImageDomain *domain) {
    iftImage *img = iftReadImageByExt(img_path);

    if (iftDictContainKey("--img-depth", args, NULL)) {
        int img_depth = iftGetLongValFromDict("--img-depth", args);
        if (img_depth <= 0)
            iftError("Invalid Image Depth: %d... Try > 0", "iftGetOptionalArgs", img_depth);

        *img_max_range = (1 << img_depth) - 1; // (2^img_depth) - 1
    }
    else {
        *img_max_range = iftNormalizationValue(iftMaximumValue(img));
    }


    if (iftDictContainKey("--sampling-percentage", args, NULL))
        *sampling_perc = iftGetDblValFromDict("--sampling-percentage", args);
    else *sampling_perc = 0.5;

    if ((*sampling_perc < 0.0) || (*sampling_perc > 1.0))
        iftError("Invalid Sampling Percentage: %f... Try [0.0, 1.0]", "iftGetOptionalArgs", sampling_perc);

    (*domain).xsize = img->xsize;
    (*domain).ysize = img->ysize;
    (*domain).zsize = img->zsize;
    iftDestroyImage(&img);

    if (iftDictContainKey("--num-threads", args, NULL)) {
        int n_threads = iftGetLongValFromDict("--num-threads", args);
        omp_set_num_threads(n_threads);
    }

    printf("- Image Range: [0, %d]\n", *img_max_range);
    printf("- Sampling Percentage: %f\n", *sampling_perc);
    if ((*domain).zsize > 1) // 3D
        printf("- Image's Domain: %dx%dx%d\n", (*domain).xsize, (*domain).ysize, (*domain).zsize);
    else printf("- Image's Domain: %dx%d\n", (*domain).xsize, (*domain).ysize);
    puts("-----------------------\n");
}


void iftBuildGlobalDataSet(iftDataSet **Zc_imgs, iftDataSet **Zt_imgs, size_t n_imgs, float sampling_perc,
                           bool balance_dataset, iftDataSet **Zc, iftDataSet **Zt) {
    iftSampler **samplers = (iftSampler**) iftAlloc(n_imgs, sizeof(iftSampler*));
    int n_total_samples   = 0;

    // gets a sampler for each dataset 
    for (int i = 0; i < n_imgs; i++) {
        int n_train_samples = (int) Zc_imgs[i]->nsamples * sampling_perc;
        n_total_samples    += Zc_imgs[i]->nsamples;

        iftIntArray *true_labels = iftGetDataSetTrueLabels(Zc_imgs[i]);
        if (balance_dataset) {
            samplers[i] = iftBalancedRandomSubsampling(true_labels->val, true_labels->n, 1, n_train_samples);
        } else {
            samplers[i] = iftStratifiedRandomSubsampling(true_labels->val, true_labels->n, 1, n_train_samples);   
        }
        iftDestroyIntArray(&true_labels);
    }

    *Zc = iftMergeDataSetArrayBySampling(Zc_imgs, n_imgs, samplers, false);
    *Zt = iftMergeDataSetArrayBySampling(Zt_imgs, n_imgs, samplers, false);
    // *Zc = iftMergeDataSetArray(Zc_imgs, n_imgs, false);
    // *Zt = iftMergeDataSetArray(Zt_imgs, n_imgs, false);
    printf("- Final Dataset: nsamples: %d/%d, nfeats: %d, nclasses: %d\n", (*Zc)->nsamples, n_total_samples, (*Zc)->nfeats, (*Zc)->nclasses);
    printf("- Final Dataset: nsamples: %d/%d, nfeats: %d, nclasses: %d\n\n", (*Zt)->nsamples, n_total_samples, (*Zt)->nfeats, (*Zt)->nclasses);

    for (int i = 0; i < n_imgs; i++)
        iftDestroySampler(&samplers[i]);
    iftFree(samplers);
}


void iftWriteDataSetAsImage(  iftDataSet *Zc,   iftDataSet *Zt, iftImageDomain domain, const char *out_base) {
    iftImage *coord_img  = iftCreateImage(domain.xsize, domain.ysize, domain.zsize);
    iftImage *bright_img = iftCreateImage(domain.xsize, domain.ysize, domain.zsize);

    for (int s = 0; s < Zc->nsamples; s++) {
        iftVoxel v;
        v.x = Zc->sample[s].feat[0];
        v.y = Zc->sample[s].feat[1];
        v.z = Zc->sample[s].feat[2];

        int p = iftGetVoxelIndex(coord_img, v);
        coord_img->val[p]  = Zc->sample[s].truelabel;
        bright_img->val[p] = Zt->sample[s].truelabel;
    }

    // printf("--> Z->nsamples: %d\n", Z->nsamples);
    char *out_path = iftConcatStrings(2, out_base, "_coords.scn");
    iftWriteImageByExt(coord_img, out_path);
    iftFree(out_path);

    out_path = iftConcatStrings(2, out_base, "_bright.scn");
    iftWriteImageByExt(bright_img, out_path);
    iftFree(out_path);

    iftDestroyImage(&coord_img);
}
/*************************************************************/