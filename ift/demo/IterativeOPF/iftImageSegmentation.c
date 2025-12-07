#include "ift.h"
#include "iftIterativeOPF.h"

/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(iftDict *args, char **input_image, int* alg, int *n_clusters, int *max_iterations, char **output_image);
/*************************************************************/

int main(int argc, const char* argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // It is mandatory to include the following line of code in order to select random initial centers.
    iftRandomSeed(time(NULL));

    // mandatory args
    char *input_image = NULL, *output_image = NULL; 
    int alg, num_clusters, max_iterations;
    timer *tic = NULL, *toc = NULL;

    iftGetRequiredArgs(args, &input_image, &alg, &num_clusters, &max_iterations, &output_image);

    iftImage *img = iftReadImageByExt(input_image);
    iftImage *label;
    iftMImage *mimg = iftImageToMImage(img, LABNorm_CSPACE);
    iftDataSet *Z = iftMImageToDataSet(mimg, NULL, 0);

    Z->ntrainsamples = Z->nsamples;

    iftSetStatus(Z, IFT_TRAIN);

    // iftKnnGraph* kgraph = iftCreateKnnGraph(Z, 10);

    iftGraph *graph = iftCreateGraph(Z);
    //graph->is_complete = true;

    float r = sqrt(2);
    iftAdjRel *A = iftCircular(r);

    iftSetMGraphAdjacencySets(graph, mimg, A);

    tic = iftTic();

    switch(alg) {
        case 0:
            iftIterativeOPF(graph, num_clusters, max_iterations, IFT_MAX, IFT_XY_COORD);
            iftFindOptimumCentersDynOPF(graph, graph->center_hist, num_clusters, IFT_MAX);
            break;
        case 1:
            iftIteratedWatersheds(graph, num_clusters, max_iterations, IFT_SUM, IFT_XY_COORD, false);
            iftFindOptimumCentersOPF(graph, graph->center_hist, num_clusters, IFT_SUM, false);
            break;
        case 2:
            iftKMeans(Z, num_clusters, max_iterations, 0.001, NULL, NULL);
            break; 
        default:
            puts("Unknown option.");
            exit(0);
    }
        
    toc = iftToc();

    label = iftDataSetClusterInformationToLabelImage(graph->Z, NULL, false);

    A = iftCircular(1.5);
    iftColorTable *cmap = iftCreateColorTable(num_clusters + 1);
    iftDrawLabels(img, label, cmap, A, true, 0.25);

    //label_img = iftNormalizeImage(out_img, A, 255);

    iftWriteImageByExt(img, output_image);
    //iftWriteImageByExt(label, output_image);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(tic, toc)));

    return 0;
}

/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Segments an input image by Iterative OPF, which uses a new schema to estimate the arc-weights.\n";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--input-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
                .required=true, .help="Input image."},
        {.short_name = "-a", .long_name = "--alg", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                .required=true, .help="Algorithm; 0) Iterative OPF, 1) Iterated Watersheds, 2) K-Means"},
        {.short_name = "-n", .long_name = "--n-clusters", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                .required=true, .help="Number of clusters."},
        {.short_name = "-m", .long_name = "--max-iterations", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                .required=true, .help="Maximum number of iterations (e.g., 20)."},
        {.short_name = "-o", .long_name = "--output-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
                .required=true, .help="Output file."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(iftDict *args, char **input_file, int *alg, int *n_clusters, int *max_iterations, char **output_file) {
    char algorithms[3][50] = {"Iterative OPF", "Iterated Watersheds", "K-Means"};

    *input_file     = iftGetStrValFromDict("--input-image", args);
    *alg            = (int) iftGetLongValFromDict("--alg", args);
    *n_clusters     = (int) iftGetLongValFromDict("--n-clusters", args);
    *max_iterations = (int) iftGetLongValFromDict("--max-iterations", args);
    *output_file    = iftGetStrValFromDict("--output-image", args);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

    puts("-----------------------");
    printf("- Input image: \"%s\"\n", *input_file);
    printf("- Algorithm: \"%s\"\n", algorithms[*alg]);
    printf("- # of clusters: \"%d\"\n", *n_clusters);
    printf("- Maximum # of iterations: \"%d\"\n", *max_iterations);
    printf("- Output image: \"%s\"\n", *output_file);
    puts("-----------------------\n");
}
/*************************************************************/