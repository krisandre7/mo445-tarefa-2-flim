#include "ift.h"

typedef enum {SLIC_PARAM_GLOBAL, SLIC_PARAM_LOCAL} SlicParamEvalType;
enum {SLIC_NSUPERVOXELS=0, SLIC_COMPACTNESS};

typedef struct ift_slic_param_optimizer {
    iftDict *label_dict;
    int img_max_range;
    bool multi_label;
    SlicParamEvalType type;
}iftSLICParamOptimizer;


/*********************************************************************/

iftDict *iftGetArgs(int argc, const char *argv[]);

void iftGetRequiredArgs(  iftDict *args, iftFileSet **img_set, iftDict **label_dict, int *input_nsupervoxels0,
                        int *input_nsupervoxels1, double *comp0, double *comp1, size_t *nsteps,
                        SlicParamEvalType *type);

void iftValidateFileSet(  iftFileSet *img_files, const char *img_entry);
void iftValidateInputs(const char *img_entry, const char *label_entry);

void iftGetOptionalArgs(  iftDict *args, const char *img_path, int *img_max_range, bool *multi_label);

/*********************************************************************/

iftIntMatrix * iftCountSupervoxelSubdivision(iftImage *supervoxels, iftImage *gt);

double iftAvgSupervoxelLabelAssignmentPerc(iftIntMatrix *subdivision_count);

double iftAvgSubdividedSupervoxelLabelAssignmentPerc(iftIntMatrix *subdivision_count, double *perc_supervoxels);

/**
 * @brief Computes SLIC supervoxels for a set of images and evaluates the accuracy of label assignment by majority vote.
 *
 * The accuracy is measured by the average number of subdivided supervoxels according to the ground truth segmentation
 * of objects of interest.
 *
 * @param img_set The input image file set.
 * @param label_dict The dictionary containing the input images as keys and the paths to the corresponding ground truths as values.
 * @param input_n_clusters The maximum number of supervoxels to be generated.
 * @param comp The compactness parameter for SLIC
 * @param img_max_range The brightness range for the image
 * @multi_label If true, then
 */
double iftEvalSlicLabelAssignmentAccuracy(  iftFileSet *img_set,   iftDict *label_dict, int input_n_supervoxels,
                                          double comp, int img_max_range, bool multi_label, SlicParamEvalType type,
                                          double *global_perc_supervoxels);

double iftGridSearchSLICParamsFunc(void* problem, iftFileSet* img_set, double *params);

void iftGridSearchSLICParamOptimization(iftFileSet *img_set, iftDict *label_dict, int img_max_range, bool multi_label,
                                        SlicParamEvalType type, int nsupervoxels0, int nsupervoxels1, double comp0,
                                        double comp1, size_t nsteps, int *best_nsupervoxels, double *best_comp);

/*********************************************************************/

int main(int argc, const char *argv[]) {
    int input_nsupervoxels0, input_nsupervoxels1;

    int img_max_range = 255;
    double comp0, comp1, global_avg = 0.0, global_perc_supervoxels = 0.0;
    size_t nsteps = 10;
    int best_nsupervoxels = 0;
    double best_comp = 0;

    bool multi_label = false;
    iftFileSet *img_set = NULL;
    iftFileSet *label_set = NULL;
    iftDict *args = NULL;
    iftDict *label_dict = NULL;
    SlicParamEvalType type = SLIC_PARAM_GLOBAL;

    /*-------------------------------*/

    int MemDinInicial, MemDinFinal;
    MemDinInicial = iftMemoryUsed();

    /*-------------------------------*/

    args = iftGetArgs(argc, argv);

    iftGetRequiredArgs(args, &img_set, &label_dict, &input_nsupervoxels0, &input_nsupervoxels1, &comp0, &comp1, &nsteps,
                       &type);
    iftGetOptionalArgs(args, img_set->files[0]->path, &img_max_range, &multi_label);


//    iftEvalSlicLabelAssignmentAccuracy(img_set, label_dict, input_nsupervoxels0, comp0, img_max_range, multi_label, type,
//                                       &global_perc_supervoxels);
    iftGridSearchSLICParamOptimization(img_set, label_dict, img_max_range, multi_label, type, input_nsupervoxels0,
                                       input_nsupervoxels1, comp0, comp1, nsteps, &best_nsupervoxels, &best_comp);
    iftDestroyFileSet(&img_set);
    iftDestroyFileSet(&label_set);
    iftDestroyDict(&args);

    printf("Global average: %.3lf. Global perc. supervoxels: %.3lf\n", global_avg, global_perc_supervoxels);


    /*-------------------------------*/

    MemDinFinal = iftMemoryUsed();
    iftVerifyMemory(MemDinInicial, MemDinFinal);

    /*-------------------------------*/
    return 0;
}



/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = "Evaluates a superpixel creation algorithm to determine the best parameters that make "\
                                        "the resulting superpixels to respect the border of ground truth object segmentation";
    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-i", .long_name = "--img-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Dir with the Input Images or a CSV file with their pathnames."},
            {.short_name = "-l", .long_name = "--label-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Dir with the Label Images or a CSV file with their pathnames."},
            {.short_name = "-k", .long_name = "--num-supervoxels", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Interval of the Number of Superpixels in which grid search must be performed "\
                                            "(e.g. \"100 1000\"."},
            {.short_name = "-c", .long_name = "--compactness", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Interval of SLIC Compactness in which grid search must be performed "\
                                            "(e.g. \"1.0 10.0\"."},
            {.short_name = "-n", .long_name = "--nsteps", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                    .required=true, .help="Number of steps to divide the parameter grid search intervals."},
            {.short_name = "-p", .long_name = "--param-eval-type", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Type of parameter evaluation score to be considered: GLOBAL or LOCAL"},
            {.short_name = "", .long_name = "--multi-label", .has_arg=false, .arg_type=IFT_BOOL_TYPE,
                    .required=false, .help="If set, the supervoxel subdivision counting takes into account each label. By default, the input label is binarized."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}



void iftGetRequiredArgs(  iftDict *args, iftFileSet **img_set, iftDict **label_dict, int *input_nsupervoxels0,
                        int *input_nsupervoxels1, double *comp0, double *comp1, size_t *nsteps,
                        SlicParamEvalType *type) {
    const char *img_entry   = iftGetConstStrValFromDict("--img-entry", args);
    const char *label_entry = iftGetConstStrValFromDict("--label-entry", args);
    const char *param_type  = iftGetConstStrValFromDict("--param-eval-type", args);
    const char *nsupervoxels = iftGetConstStrValFromDict("--num-supervoxels", args);
    const char *compactness  = iftGetConstStrValFromDict("--compactness", args);

    *nsteps = iftGetLongValFromDict("--nsteps", args);

    if(sscanf(nsupervoxels, "%d %d", input_nsupervoxels0, input_nsupervoxels1) != 2)
        iftError("Please pass an interval of supervoxels such as \"100 1000\" to perform grid search", "iftGetRequiredArgs");

    if(sscanf(compactness, "%lf %lf", comp0, comp1) != 2)
        iftError("Please pass an interval of compactness such as \"1 10\" to perform grid search", "iftGetRequiredArgs");

    iftValidateInputs(img_entry, label_entry);

    if(iftCompareStrings(param_type, "LOCAL"))
        *type = SLIC_PARAM_LOCAL;
    else if (iftCompareStrings(param_type, "GLOBAL"))
        *type = SLIC_PARAM_GLOBAL;
    else
        iftError("Please choose between GLOBAL and LOCAL supervoxel parameter evaluation type!", "iftGetRequiredArgs");

    *img_set              = iftLoadFileSetFromDirOrCSV(img_entry, 0, true);
    iftFileSet *label_set = iftLoadFileSetFromDirOrCSV(label_entry, 0, true);

    iftValidateFileSet(*img_set, img_entry);
    iftValidateFileSet(label_set, label_entry);

    *label_dict = iftFileSetToDict(label_set);
    iftDestroyFileSet(&label_set);


    puts("-----------------------");
    printf("- Image Entry: \"%s\"\n", img_entry);
    printf("- Label Image Entry: \"%s\"\n", label_entry);
    printf("- Interval of Desired Number of Superpixels/Supervoxels: %d %d\n", *input_nsupervoxels0, *input_nsupervoxels1);
    printf("- Compactness interval: %lf %lf\n", *comp0, *comp1);
    printf("- Parameter Evaluation Type: %s\n", param_type);
    puts("-----------------------");
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

void iftValidateInputs(const char *img_entry, const char *label_entry) {
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
}


void iftGetOptionalArgs(  iftDict *args, const char *img_path, int *img_max_range, bool *multi_label) {
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


    *multi_label = iftDictContainKey("--multi-label", args, NULL);
}

iftIntMatrix *iftCountSupervoxelSubdivision(iftImage *supervoxels, iftImage *gt) {
    iftIntMatrix *subdivision_count = NULL;
    iftIntArray  *labels = NULL;
    int nlabels, nsupervoxels;

    if(iftMinimumValue(supervoxels) != 1)
        iftError("We expect the supervoxels to be labeled from 1 to n!", "iftCountSupervoxelSubdivision");

    nlabels = iftMaximumValue(gt)+1;
    nsupervoxels = iftMaximumValue(supervoxels);

    subdivision_count = iftCreateIntMatrix(nlabels, nsupervoxels);
    labels      = iftCreateIntArray(nlabels);

    for(int p = 0; p < supervoxels->n; p++) {
        int lb, c;

        lb = gt->val[p];
        c = supervoxels->val[p];

        iftMatrixElem(subdivision_count, lb, c - 1)++;
    }

    // Sorting the number of labels for each supervoxel in descending order in-place, for posterior determination
    // of the average global percentage of labeled pixel frequency used to assign labels to supervoxels
    for(int c = 0; c < nsupervoxels; c++) {
        iftQuickSort(iftMatrixRowPointer(subdivision_count, c), labels->val,
                     0, nlabels - 1, IFT_DECREASING);
    }

    iftDestroyIntArray(&labels);

    return subdivision_count;
}


double iftAvgSupervoxelLabelAssignmentPerc(iftIntMatrix *subdivision_count) {
    double sum = 0.0;
    size_t ndivided = 0;

    for(int c = 0; c < subdivision_count->nrows; c++) {
        int nvoxels = 0;

        for(int lb = 0; lb < subdivision_count->ncols; lb++) {
            nvoxels += iftMatrixElem(subdivision_count, lb, c);
        }

        // Summing the percentage of pixels belonging to the most frequent label under the current supervoxel to
        // determine the global average of supervoxel label assignment
        sum += iftMatrixElem(subdivision_count, 0, c) / (double)nvoxels;
        ndivided++;
    }

    return sum / ndivided;
}


double iftAvgSubdividedSupervoxelLabelAssignmentPerc(iftIntMatrix *subdivision_count, double *perc_supervoxels) {
    double sum = 0.0;
    size_t ndivided = 0;

    for(int c = 0; c < subdivision_count->nrows; c++) {
        int nvoxels = 0;

        for(int lb = 0; lb < subdivision_count->ncols; lb++) {
            nvoxels += iftMatrixElem(subdivision_count, lb, c);
        }

        if(iftMatrixElem(subdivision_count, 0, c) != nvoxels) {
            // Summing the percentage of pixels belonging to the most frequent label under the current supervoxel to
            // determine the global average of supervoxel label assignment
            sum += iftMatrixElem(subdivision_count, 0, c) / (double) nvoxels;
            ndivided++;
        }
    }

    *perc_supervoxels = ndivided / (double)subdivision_count->nrows;

    return sum / ndivided;
}


double iftEvalSlicLabelAssignmentAccuracy(  iftFileSet *img_set,   iftDict *label_dict, int input_n_supervoxels,
                                          double comp, int img_max_range, bool multi_label, SlicParamEvalType type,
                                          double *global_perc_supervoxels) {
    double global_avg = 0.0;

    if(global_perc_supervoxels != NULL)
        *global_perc_supervoxels = 0.0;

    for(int i = 0; i < img_set->n; i++) {
        double avg = 0.0, perc_supervoxels = 0.0;

        const char *img_path   = img_set->files[i]->path;
        char *img_key          = iftFileKey(img_path);
        const char *label_path = iftGetConstStrValFromDict(img_key, label_dict);

        iftFree(img_key);

        iftImage *img = iftReadImageByExt(img_path);
        iftImage *label = iftReadImageByExt(label_path);
        iftImage *gt = NULL;
        iftImage *supervoxels = iftGenerateSuperpixelsBySlic(img, NULL, input_n_supervoxels, comp, img_max_range, NULL);

        if(multi_label) {
            gt = label;
        } else {
            gt = iftThreshold(label, 1, IFT_INFINITY_INT, 1);
        }

        iftIntMatrix *subdivision_count = iftCountSupervoxelSubdivision(supervoxels, gt);

        if(type == SLIC_PARAM_LOCAL) {
            avg = iftAvgSubdividedSupervoxelLabelAssignmentPerc(subdivision_count, &perc_supervoxels);
        } else {
            avg = iftAvgSupervoxelLabelAssignmentPerc(subdivision_count);
        }

        global_avg += avg;
        if(global_perc_supervoxels != NULL)
            *global_perc_supervoxels += perc_supervoxels;

//        printf("Average supervoxel subdivision_count for image %-16s: %.3lf. Perc of supervoxels: %.3lf\n", img_path, avg, perc_supervoxels);

//        iftAdjRel *A = iftCircular(1.0);
//        iftAdjRel *B = iftCircular(0.0);
//        iftImage *result = iftCopyImage(img);
//        iftDrawBorders(result, supervoxels, A, iftRGBtoYCbCr(iftRGBColor(255,0,0), 255), B);
//        char *bname = iftFilename(img_path, iftFileExt(img_path));
//        iftWriteImageByExt(supervoxels, "%s_supervoxels.pgm", bname);
//        iftWriteImageByExt(result, "%s_result.ppm",bname);
//        iftFree(bname);
//        iftDestroyAdjRel(&A);
//        iftDestroyAdjRel(&B);
//        iftDestroyImage(&result);

        // Destroyers
        if(gt != label) {
            iftDestroyImage(&gt);
        }

        iftDestroyImage(&label);
        iftDestroyImage(&img);
        iftDestroyImage(&supervoxels);
        iftDestroyIntMatrix(&subdivision_count);
    }


    global_avg /= iftMax(img_set->n, 1);
    if(global_perc_supervoxels != NULL) {
        if (type == SLIC_PARAM_GLOBAL) {
            *global_perc_supervoxels = 1.0;
        } else {
            *global_perc_supervoxels /= iftMax(img_set->n, 1);
        }
    }

    return global_avg;
}

double iftGridSearchSLICParamsFunc(void* problem, iftFileSet* img_set, double *params) {
    iftSLICParamOptimizer *context = (iftSLICParamOptimizer*)problem;

    iftDict *label_dict = context->label_dict;
    int img_max_range   = context->img_max_range;
    bool multi_label    = context->multi_label;
    SlicParamEvalType type = context->type;
    double score, global_avg, global_perc_supervoxels = 0.0;

    /* Getting current parameters */
    int nsupervoxels    = iftRound(params[SLIC_NSUPERVOXELS]);
    double comp         = params[SLIC_COMPACTNESS];

    fprintf(stderr, "Current optimization values. Number of supervoxels: %d. Compactness: %.3lf.\n", nsupervoxels, comp);
    global_avg = iftEvalSlicLabelAssignmentAccuracy(img_set, label_dict, nsupervoxels, comp, img_max_range, multi_label,
                                                    type, &global_perc_supervoxels);

    if(type == SLIC_PARAM_GLOBAL) {
        score = global_avg;
        fprintf(stderr, "Score: %.3lf.\n", score);
    } else {
        // We aim to minimize the number of subdivided voxels while ensuring that the divided ones are assigned as much
        // as possible to a single label. Hence, we multiply the complement of the number of divided voxels by the
        // global average
        score = (1.0 - global_perc_supervoxels)*global_avg;
        fprintf(stderr, "Score: %.3lf. Global percentage of subdivided supervoxels: %.3lf\n", score, global_perc_supervoxels);
    }



    return score;
}


void iftGridSearchSLICParamOptimization(iftFileSet *img_set, iftDict *label_dict, int img_max_range, bool multi_label,
                                        SlicParamEvalType type, int nsupervoxels0, int nsupervoxels1, double comp0,
                                        double comp1, size_t nsteps, int *best_nsupervoxels, double *best_comp) {
    iftParamOptimizer *opt = NULL;
    iftSLICParamOptimizer *context = NULL;
    double step;

    context = (iftSLICParamOptimizer*)iftAlloc(1, sizeof(iftSLICParamOptimizer));
    context->label_dict = label_dict;
    context->img_max_range = img_max_range;
    context->multi_label   = multi_label;
    context->type          = type;

    opt = iftCreateParamOptimizer(2);

    /* Initializing search space */
    step = iftMax((nsupervoxels1 - nsupervoxels0)/nsteps, 1);
    opt->paramsSpace[SLIC_NSUPERVOXELS] = iftUniformSearchSpace(nsupervoxels0, nsupervoxels1, step);
    step = iftMax((comp1 - comp0)/nsteps, 1);
    opt->paramsSpace[SLIC_COMPACTNESS]  = iftUniformSearchSpace(comp0, comp1, step);

    /* Searching for the best parameters */
    iftGridSearchDescriptor(opt, iftGridSearchSLICParamsFunc, img_set, context);

    *best_nsupervoxels = iftRound(opt->params[SLIC_NSUPERVOXELS]);
    *best_comp         = opt->params[SLIC_COMPACTNESS];

    fprintf(stderr, "Best SLIC parameters. Number of supervoxels: %d. Compactnes: %.3lf. Score %.3lf\n",
            *best_nsupervoxels, *best_comp, opt->score);

    iftFree(context);
    iftDestroyParamOptimizer(&opt);
}