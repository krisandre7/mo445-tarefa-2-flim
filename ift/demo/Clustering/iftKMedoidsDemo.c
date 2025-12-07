#include "ift.h"

/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(iftDict *args, char **input_file, int* n_clusters, int *max_iter);
/*************************************************************/

int main(int argc, const char* argv[]) {

    iftRandomSeed(time(NULL));

    iftDict *args = iftGetArgs(argc, argv);

    char  *input_file = NULL; 
    int   n_clusters, max_iter;
    timer *tic = NULL, *toc = NULL;

    iftGetRequiredArgs(args, &input_file, &n_clusters, &max_iter);

    iftDataSet *Z = iftReadCSVDataSet(input_file, ',', 1, NULL);

    tic = iftTic();

    iftKMedoids(Z, n_clusters, max_iter, iftEuclideanDistance);

    toc = iftToc();

    int *prototypes = iftGetDataSetPrototypes(Z);
    int *num_elem_cluster = iftAllocIntArray(n_clusters);

    for (int i = 0; i < Z->nsamples; i++) {
        num_elem_cluster[Z->sample[i].group - 1] += 1;
    }

    for (int i = 0; i < n_clusters; i++) {
        printf("###########\n");
        printf("Cluster #%d\n", i + 1);
        printf("\tPrototype: %d\n\t\t", prototypes[i]);
        iftPrintFloatArray(Z->sample[prototypes[i]].feat, Z->nfeats);
        printf("# of elements: %d\n", num_elem_cluster[i]);
        printf("###########\n");
    }

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
                .required=true, .help="Number of iterations."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(iftDict *args, char **input_file, int* n_clusters, int *max_iter) {
    *input_file = iftGetStrValFromDict("--input-file", args);
    *n_clusters = (int) iftGetLongValFromDict("--n-clusters", args);
    *max_iter   = (int) iftGetLongValFromDict("--max-iter", args);

    puts("-----------------------");
    printf("- Input file: \"%s\"\n", *input_file);
    printf("- # of clusters: %d\n", *n_clusters);
    printf("- # of iterations: %d\n", *max_iter);
    puts("-----------------------\n");
}
/*************************************************************/
