/**
 * @file
 * @brief Segments an image with SLIC and SVM.
 * The program segments an image by generating supervoxels for that, and classifying using OPF classifier. \n
 * Two OPF classifiers are trained: one for 3D coordinates from the supervoxel's center voxel ([x, y, z]), and other
 * for brightness values of the supervoxel ([avg, stdev, min, max]).
 * @note See the source code in @ref iftSegmentImageBySLIC-SVM.c
 *
 * @example iftSegmentImageBySLIC-SVM.c
 * @brief Segments an image with SLIC and SVM.
 * The program segments an image by generating supervoxels for that, and classifying using OPF classifier. \n
 * Two OPF classifiers are trained: one for 3D coordinates from the supervoxel's center voxel ([x, y, z]), and other
 * for brightness values of the supervoxel ([avg, stdev, min, max]).
 * @author Samuel Martins
 * @date Jun 19, 2016
 */


#include "ift.h"


/************************** HEADERS **************************/
typedef enum {
    COORD_FEATS,
    BRIGHTNESS_FEATS
} iftSupervoxelFeats;

iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, char **coord_data_path, char **bright_data_path,
                        int *input_n_clusters, double *comp, char **out_base);
void iftValidateInputs(const char *img_path, const char *coord_data_path, const char *bright_data_path,
                       int input_n_clusters, double comp, const char *out_base);
void iftGetOptionalArgs(  iftDict *args, double *C);
int iftGetImageMaxRange(  iftDict *args,   iftImage *img);
iftDataSet *iftClassifySupervoxelsBySVM(iftSupervoxelFeats feat_type,   iftImage *img,   iftImage *super_img,
                                                  const char *data_path, double C, int img_max_range);
iftImage *iftBuildLabelImageFromDataSet(  iftImage *super_img,   iftDataSet *Zimg);
iftImage *iftBuildFuzzyImageFromDataSet(  iftImage *super_img,   iftDataSet *Zimg, int img_max_range);
/*************************************************************/


int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *img_path          = NULL;
    char *coord_data_path   = NULL;
    char *bright_data_path = NULL;
    int input_n_clusters;
    double comp;
    char *out_base;
    // optional args
    double C;

    iftGetRequiredArgs(args, &img_path, &coord_data_path, &bright_data_path, &input_n_clusters, &comp, &out_base);
    iftGetOptionalArgs(args, &C);

    timer *t1 = iftTic();

    iftImage *img     = iftReadImageByExt(img_path);
    int img_max_range = iftGetImageMaxRange(args, img);

    puts("- Generating Supervoxels by SLIC");
    iftImage *super_img = iftGenerateSuperpixelsBySlic(img, NULL, input_n_clusters, comp, img_max_range, NULL);
    iftWriteSuperpixelBorders(img, super_img, "super.zip");
    iftWriteImage(super_img, "super_img.scn");

    // puts("- Classifying by Coordinates OPF");
    // iftDataSet *Zc_img = iftClassifySupervoxelsBySVM(COORD_FEATS, NULL, super_img, coord_data_path, img_max_range);
    // puts("- Getting Coord Dataset's Image");
    // iftImage *coord_img = iftBuildLabelImageFromDataSet(super_img, Zc_img);
    // iftWriteImageByExt(coord_img, "out/coord_img.scn");
    // iftDestroyImage(&coord_img);

    puts("\n- Classifying by Brightness OPF");
    iftDataSet *Zb_img = iftClassifySupervoxelsBySVM(BRIGHTNESS_FEATS, img, super_img, bright_data_path, C, img_max_range);
    puts("- Getting Brightness Dataset's Image");
    iftImage *bright_img = iftBuildLabelImageFromDataSet(super_img, Zb_img);
    iftWriteImageByExt(bright_img, "out/bright_img.scn");
    iftDestroyImage(&bright_img);

    // puts("- Getting Fuzzy Dataset's Image");
    // iftImage *fuzzy = iftBuildFuzzyImageFromDataSet(super_img, Zb_img, img_max_range);
    // iftWriteImageByExt(fuzzy, "out/fuzzy.scn");
    // iftDestroyImage(&fuzzy);


    iftDestroyImage(&super_img);
    // iftDestroyDataSet(&Zc_img);
    iftDestroyDataSet(&Zb_img);



    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyImage(&img);
    iftFree(img_path);
    iftFree(coord_data_path);
    iftFree(bright_data_path);
    iftFree(out_base);

    return 0;
}



/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "This programs segments an image with SLIC and SVM.\n" \
        "It segments an image by generating supervoxels, and classifying them using OPF classifier.\n" \
        "There are two options for supervoxel's feature vector:\n" \
        "(1) COORDS: 3D coordinates from the supervoxel's center voxel: [x, y, z]\n" \
        "(2) BRIGHT: Brightness values of the supervoxel: [avg, stdev, min, max]";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--input-test-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Image to be segmented."},
        {.short_name = "-d", .long_name = "--train-coord-dataset", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the Dataset of Coordinates."},
        {.short_name = "-b", .long_name = "--train-brightness-dataset", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname from the Dataset of Coordinates."},
        {.short_name = "-k", .long_name = "--num-clusters", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=true, .help="Approximate Number of Superpixels."},
        {.short_name = "-c", .long_name = "--compactness", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=true, .help="Compactness: Relative Importance of color similarity and spatial proximity.\n" \
                                "Try values on a log scale (0.01, 0.1, 1, 10, 100) before refining " \
                                "around a chosen value."},
        {.short_name = "-o", .long_name = "--output-basename", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output basename for the Output Segmented Images."},
        {.short_name = "-C", .long_name = "--svm-penalty-param", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Penalty parameter C of the error term.\n" \
                               "Default: 1.0"},
        {.short_name = "-j", .long_name = "--img-depth", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=false, .help="Input Image Depth in bits (8, 12, 16, ...)\n" \
                                "Default: it tries to find the image depth automatically from the First Image"}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **img_path, char **coord_data_path, char **bright_data_path,
                        int *input_n_clusters, double *comp, char **out_base) {
    *img_path         = iftGetStrValFromDict("--input-test-img", args);
    *coord_data_path  = iftGetStrValFromDict("--train-coord-dataset", args);
    *bright_data_path = iftGetStrValFromDict("--train-brightness-dataset", args);
    *input_n_clusters = iftGetLongValFromDict("--num-clusters", args);
    *comp             = iftGetDblValFromDict("--compactness", args);
    *out_base         = iftGetStrValFromDict("--output-basename", args);

    iftValidateInputs(*img_path, *coord_data_path, *bright_data_path, *input_n_clusters, *comp, *out_base);

    puts("-----------------------");
    printf("- Input Image: \"%s\"\n", *img_path);
    printf("- Coord. Dataset: \"%s\"\n", *coord_data_path);
    printf("- Brightness Dataset: \"%s\"\n", *bright_data_path);
    printf("- Input Desired Number of Superpixels/Supervoxels: %d\n", *input_n_clusters);
    printf("- Compactness: %lf\n", *comp);
    printf("- Output Image Basename: \"%s\"\n", *out_base);
    puts("-----------------------");
}



void iftValidateInputs(const char *img_path, const char *coord_data_path, const char *bright_data_path,
                       int input_n_clusters, double comp, const char *out_base) {
    // input image
    if (!iftIsImageFile(img_path))
        iftError("The input image \"%s\" is not a valid image (*.[pgm,ppm,png,scn]) or it does not exist",
                 "iftValidateInputs", img_path);

    // coord. dataset
    if (!iftFileExists(coord_data_path))
        iftError("Coord. Dataset does not exist: \"%s\"", "iftValidateInputs", coord_data_path);
    else if (!iftEndsWith(coord_data_path, ".data"))
        iftError("Invalid dataset extension for the Coord. Dataset: \"%s\"", "iftValidateInputs", coord_data_path);

    // bright. dataset
    if (!iftFileExists(bright_data_path))
        iftError("Brightness Dataset does not exist: \"%s\"", "iftValidateInputs", bright_data_path);
    else if (!iftEndsWith(bright_data_path, ".data"))
        iftError("Invalid dataset extension for the Brightness Dataset: \"%s\"", "iftValidateInputs", bright_data_path);

    // bright. dataset
    if (!iftFileExists(bright_data_path))
        iftError("Brightness Dataset does not exist: \"%s\"", "iftValidateInputs", bright_data_path);
    else if (!iftEndsWith(bright_data_path, ".data"))
        iftError("Invalid dataset extension for the Brightness Dataset: \"%s\"", "iftValidateInputs", bright_data_path);

    // num of clusters
    if (input_n_clusters <= 0)
        iftError("Invalid Desired Number of Superpixels: %ld... Try > 0", "iftValidateInputs", input_n_clusters);

    // compactness
    if (comp <= 0.0)
        iftError("Invalid Compactness: %lf... Try > 0 (0.01, 0.1, 1, 10)", "iftValidateInputs", comp);

    // // output image
    // if (!iftIsImagePathnameValid(out_base))
    //     iftError("Invalid Output Image Pathname: \"%s\"", "iftValidateInputs", out_base);

    char *parent_dir = iftParentDir(out_base);
    if (!iftDirExists(parent_dir)) {
        iftMakeDir(parent_dir);
    }
    iftFree(parent_dir);
}


int iftGetImageMaxRange(  iftDict *args,   iftImage *img) {
    int img_max_range;

    if (iftDictContainKey("--img-depth", args, NULL)) {
        int img_depth = iftGetLongValFromDict("--img-depth", args);
        if (img_depth <= 0)
            iftError("Invalid Image Depth: %d... Try > 0", "iftGetOptionalArgs", img_depth);

        img_max_range = (1 << img_depth) - 1; // (2^img_depth) - 1
    }
    else img_max_range = iftNormalizationValue(iftMaximumValue(img));

    printf("- Image Range: [0, %d]\n", img_max_range);
    puts("-----------------------\n");

    return img_max_range;
}


void iftGetOptionalArgs(  iftDict *args, double *C) {
    if (iftDictContainKey("-C", args, NULL)) {
        *C = iftGetDblValFromDict("-C", args);

        if (*C <= 0)
            iftError("Invalid SVM Penaulty Prameter C: %lf <= 0", "iftGetOptionalArgs", *C);
    }
    else *C = 1.0;

    printf("- SVM C: %lf\n", *C);
}


iftDataSet *iftClassifySupervoxelsBySVM(iftSupervoxelFeats feat_type,   iftImage *img,   iftImage *super_img,
                                                  const char *data_path, double C, int img_max_range) {
    puts("  - Reading Train Dataset");
    iftDataSet *Ztrain      = iftReadOPFDataSet(data_path);
    printf("    - nsamples: %d, nfeats: %d, nclasses: %d\n", Ztrain->nsamples, Ztrain->nfeats, Ztrain->nclasses);
    iftSetStatus(Ztrain, IFT_TRAIN);
    iftDataSet *Ztrain_norm = iftNormalizeDataSet(Ztrain);
    iftDestroyDataSet(&Ztrain);
    Ztrain = Ztrain_norm;

    puts("  - Reading Test Dataset");
    iftDataSet *Ztest      = NULL;
    if (feat_type == COORD_FEATS) {
        puts("    - COORDINATE FEATS");
        Ztest = iftSupervoxelsToVoxelCoordsDataSet(super_img, false, NULL, false);
    }
    else {
        puts("    - BRIGHTNESS FEATS");
        // Ztest = iftSupervoxelsToGrayAvgStdevMinMaxDataSet(img, super_img, img_max_range, false, NULL);
        Ztest = iftSupervoxelsToHistogramDataSet(img, super_img, 256, NULL);
    }
    printf("    - nsamples: %d, nfeats: %d\n", Ztest->nsamples, Ztest->nfeats);

    iftDataSet *Ztest_norm = iftNormalizeTestDataSet(Ztrain_norm, Ztest);
    iftDestroyDataSet(&Ztest);
    Ztest           = Ztest_norm;
    Ztest->nclasses = Ztrain->nclasses;
    
    puts("  - Training SVM");
    iftSVM *svm = iftCreatePreCompSVC(C);
    uchar traceNormalize = 0;
    float ktrace;

    printf("- Kernealizing DataSet\n");
    iftDataSet *Zaux = iftKernelizeDataSet(Ztrain, Ztrain, LINEAR, traceNormalize, &ktrace);
    iftDestroyDataSet(&Ztrain);
    Ztrain = Zaux;
    iftSVMTrainOVA(svm, Ztrain, NULL);
    printf("- Classifying Linear SVM\n");
    iftSVMLinearClassifyOVA(svm, Ztest, IFT_TEST, NULL); // Classification
    iftDestroySVM(svm);

    #pragma omp parallel for
    for (int s = 0; s < Ztest->nsamples; s++) {
        Ztest->sample[s].label--; // decrements the predict label, because the bg, e.g., has label 1 and should have value 0
    }

    iftDestroyDataSet(&Ztrain);

    return Ztest;
}


iftImage *iftBuildLabelImageFromDataSet(  iftImage *super_img,   iftDataSet *Zimg) {
    iftImage *data_img = iftCreateImage(super_img->xsize, super_img->ysize, super_img->zsize);

    for (int p = 0; p < super_img->n; p++) {
        int c            = super_img->val[p] - 1;     // ex: supervoxel's label 2 has index [1] 
        data_img->val[p] = Zimg->sample[c].label;
    }

    return data_img;
}


iftImage *iftBuildFuzzyImageFromDataSet(  iftImage *super_img,   iftDataSet *Zimg, int img_max_range) {
    iftFImage *fimg = iftCreateFImage(super_img->xsize, super_img->ysize, super_img->zsize);

    for (int p = 0; p < super_img->n; p++) {
        int c             = super_img->val[p] - 1;     // ex: supervoxel's label 2 has index [1] 
        if (Zimg->sample[c].label >= 1) // excludes bg 
            fimg->val[p] = Zimg->sample[c].weight;
    }

    iftImage *fuzzy_img = iftFImageToImage(fimg, img_max_range);
    iftDestroyFImage(&fimg);

    return fuzzy_img;
}





