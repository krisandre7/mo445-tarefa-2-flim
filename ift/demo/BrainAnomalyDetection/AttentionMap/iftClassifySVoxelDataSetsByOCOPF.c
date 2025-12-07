#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **svoxel_img_path, char **svoxel_dataset_entry,
                  char **out_path);
void iftGetOptionalArgs(  iftDict *args, float *kmax_perc, float *quantile_of_k);
/*************************************************************/


int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    char *svoxel_img_path = NULL;
    char *svoxel_dataset_entry = NULL;
    char *out_path = NULL;
    // optional args
    float kmax_perc = 0.15;
    float quantile_of_k = 0.5;

    iftGetRequiredArgs(args, &svoxel_img_path, &svoxel_dataset_entry, &out_path);
    iftGetOptionalArgs(args, &kmax_perc, &quantile_of_k);

    timer *t1 = iftTic();

    iftImage *svoxels_img = iftReadImageByExt(svoxel_img_path);
    iftFileSet *svoxel_dataset_paths = iftLoadFileSetFromDirOrCSV(svoxel_dataset_entry, 0, true);
    iftImage *abnormal_svoxels_img = iftCreateImageFromImage(svoxels_img);

    #pragma omp parallel for
    for (int f = 0; f < svoxel_dataset_paths->n; f++) {
        const char *dataset_path = svoxel_dataset_paths->files[f]->path;
        
        char *filename = iftFilename(dataset_path, iftFileExt(dataset_path));
        int svoxel_label = atoi(iftSplitStringAt(dataset_path, "_", -1));
        iftFree(filename);

        iftDataSet *Z = iftReadDataSet(svoxel_dataset_paths->files[f]->path);
        iftDataSet *Ztrain = NULL, *Ztest = NULL;
        iftSplitDataSetAt(Z, Z->nsamples - 1, &Ztrain, &Ztest);
        iftDestroyDataSet(&Z);

        iftSetStatus(Ztrain, IFT_TRAIN);
        iftSetStatus(Ztest, IFT_TEST);

        int kmax = iftMax(iftRound(kmax_perc * Ztrain->ntrainsamples), 1);
        iftKnnGraph *graph = iftCreateKnnGraph(Ztrain, kmax);
        iftUnsupTrain(graph, iftNormalizedCut);
        printf("kmax = %d, graph-k: %d\n", kmax, graph->kmax);

        iftIntArray *groupsize = iftCreateIntArray(Ztrain->ngroups);
        for (int s = 0; s < Ztrain->nsamples; s++)
            groupsize->val[Ztrain->sample[s].group - 1]++;

        char msg[1024];
        sprintf(msg, "SVoxel: %d\n", svoxel_label);
        for (int i = 0; i < groupsize->n; i++)
            sprintf(msg, "%sGroup %d has %d samples\n", msg, i + 1, groupsize->val[i]);
        iftDestroyIntArray(&groupsize);

        iftClassifyByOneClassOPF(Ztest, graph, quantile_of_k);

        if (Ztest->sample[0].label == IFT_NEGATIVE_CLASS) {
            sprintf(msg, "%s    ====> Abnormal\n", msg);
            for (int p = 0; p < svoxels_img->n; p++) {
                if (svoxels_img->val[p] == svoxel_label) {
                    abnormal_svoxels_img->val[p] = svoxel_label;
                }
            }
        }
        puts(msg);
        puts("");

        iftDestroyDataSet(&Ztrain);
        iftDestroyDataSet(&Ztest);
        iftDestroyKnnGraph(&graph);
    }

    puts("- Writing Anomalous SVoxels");
    iftWriteImageByExt(abnormal_svoxels_img, out_path);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyImage(&svoxels_img);
    iftDestroyImage(&abnormal_svoxels_img);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] =
        "- Classify Supervoxels by OneClass OPF.\n";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-s", .long_name = "--supervoxels", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Supervoxel Image."},
        {.short_name = "-d", .long_name = "--supervoxels-datasets-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="CSV or Directory with the Supervoxel Datasets"},
        {.short_name = "-o", .long_name = "--output-anomalous-supervoxels-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Image with anomalous supervoxels."},
        {.short_name = "-k", .long_name = "--kmax-perc", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=true, .help="Percentage of the training sample used to define the OPF KMax. Default: 0.15."},
        {.short_name = "-q", .long_name = "--quantile-of-k", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=true, .help="Quantile used to define the Rank of K for the outlier thresholding. Default: 0.5 (median)"}, 
    };

    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}

void iftGetRequiredArgs(  iftDict *args, char **svoxel_img_path, char **svoxel_dataset_entry,
                  char **out_path) {
    *svoxel_img_path = iftGetStrValFromDict("--supervoxels", args);
    *svoxel_dataset_entry = iftGetStrValFromDict("--supervoxels-datasets-entry", args);
    *out_path = iftGetStrValFromDict("--output-anomalous-supervoxels-img", args);

    puts("--------------------------------");
    printf("- Supervoxel Image: %s\n", *svoxel_img_path);
    printf("- Supervoxel DataSet Entry: %s\n", *svoxel_dataset_entry);
    printf("- Output Anomalous Supervoxels: %s\n", *out_path);
    puts("--------------------------------");
}

void iftGetOptionalArgs(  iftDict *args, float *kmax_perc, float *quantile_of_k) {
    if (iftDictContainKey("--kmax-perc", args, NULL)) {
        *kmax_perc = iftGetDblValFromDict("--kmax-perc", args);
    }
    else { *kmax_perc = 0.15; }

    if (iftDictContainKey("--quantile-of-k", args, NULL)) {
        *quantile_of_k = iftGetDblValFromDict("--quantile-of-k", args);
    }
    else { *quantile_of_k = 0.5; }

    printf("- KMax Perc: %f\n", *kmax_perc);
    printf("- Quantile of K: %f\n", *quantile_of_k);
    puts("--------------------------------\n");
}
