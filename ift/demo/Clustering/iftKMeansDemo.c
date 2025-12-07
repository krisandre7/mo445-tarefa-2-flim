#include "ift.h"

/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(iftDict *args, char **input_file, int* n_clusters, int *max_iter, float *tol);
/*************************************************************/

int main(int argc, const char* argv[]) {

    iftRandomSeed(time(NULL));

    iftDict *args = iftGetArgs(argc, argv);

    char  *input_file = NULL; 
    int   n_clusters, max_iter;
    float tol;
    timer *tic = NULL, *toc = NULL;

    iftGetRequiredArgs(args, &input_file, &n_clusters, &max_iter, &tol);

    iftDataSet *Z = iftReadCSVDataSet(input_file, ',', 1, NULL);
    
    iftCentroidsHist *cent_hist = iftAlloc(1, sizeof(*cent_hist));

    // Just in case initial centroids are chosen a priori
    /*int *init_centroids = iftAllocIntArray(n_clusters);
    for (int i = 0; i < n_clusters; i++) {
        init_centroids[i] = desired sample index;
    }*/

    tic = iftTic();

    // cent_hist is an optional parameter, it records the 
    iftKMeans(Z, n_clusters, max_iter, tol, NULL, cent_hist, iftEuclideanDistance);
    // Just in case initial centroids are chosen a priori
    // iftKMeans(Z, n_clusters, max_iter, tol, init_centroids, cent_hist, iftEuclideanDistance);

    toc = iftToc();

    int *prototypes = iftGetDataSetPrototypes(Z);

    for (int i = 0; i < cent_hist->n_clusters; i++) {
        printf("###########\n");
        printf("Cluster #%d\n", i + 1);
        printf("\tCentroid:\n\t\t");
        iftPrintFloatArray(cent_hist->centroid[i]->feat, cent_hist->n_feats);
        printf("\tPrototype:\n\t\t");
        iftPrintFloatArray(Z->sample[prototypes[i]].feat, Z->nfeats);

        printf("\tCentroid history:\n");
        for (int j = 0; j < cent_hist->centroid[i]->n_hist; j++) {
            printf("\t\t");
            iftPrintFloatArray(cent_hist->centroid[i]->hist_feat[j], cent_hist->n_feats);
        }
        printf("###########\n");
    }

    printf("Total number of iterations %d\n", cent_hist->n_iter);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(tic, toc)));

    return 0;
}

/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- KMeans algorithm.\n";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--input-file", .has_arg=true, .arg_type=IFT_STR_TYPE,
                .required=true, .help="Input file (.csv)."},
        {.short_name = "-n", .long_name = "--n-clusters", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                .required=true, .help="Number of clusters."},
        {.short_name = "-m", .long_name = "--max-iter", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                .required=true, .help="Number of iterations."},
        {.short_name = "-t", .long_name = "--tol", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                .required=true, .help="Tolerance (e.g. 0.001)."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(iftDict *args, char **input_file, int* n_clusters, int *max_iter, float *tol) {
    *input_file = iftGetStrValFromDict("--input-file", args);
    *n_clusters = (int) iftGetLongValFromDict("--n-clusters", args);
    *max_iter   = (int) iftGetLongValFromDict("--max-iter", args);
    *tol        = (float) iftGetDblValFromDict("--tol", args);

    puts("-----------------------");
    printf("- Input file: \"%s\"\n", *input_file);
    printf("- # of clusters: %d\n", *n_clusters);
    printf("- # of iterations: %d\n", *max_iter);
    printf("- tolerance: %g\n", *tol);
    puts("-----------------------\n");
}
/*************************************************************/
