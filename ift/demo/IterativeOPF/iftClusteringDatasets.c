#include "ift.h"
#include "iftIterativeOPF.h"

void iftWriteCSVDataSetClustering(iftDataSet* dataset, const char* filename) {
    iftCSV* csv = iftCreateCSV(dataset->nsamples, 1);

    char buffer[IFT_STR_DEFAULT_SIZE];

    for (int s = 0; s < dataset->nsamples; ++s) {
        sprintf(buffer, "%d", dataset->sample[s].group);
        strcpy(csv->data[s][0], iftCopyString(buffer));
    }

    iftWriteCSV(csv, filename, ',');
    iftDestroyCSV(&csv);
}

int main(int argc, const char* argv[]) {
    int initial_seed = time(NULL);

    // Parameters: 
    // argv[1]: csv filename
    // argv[2]: number of clusters
    // argv[3]: max iterations
    // argv[4]: data name

    int rep = 20;

    int num_clusters = atoi(argv[2]);
    int max_iterations = atoi(argv[3]);

    char filename[128];
    char aux[128];
    char num[8];

    iftDataSet *Z = iftReadCSVDataSet(argv[1], ',', 1, NULL);
    Z->ntrainsamples = Z->nsamples;
    iftSetStatus(Z, IFT_TRAIN);
    /* iftDataSet *Z_temp= iftNormalizeDataSetByZScore(Z,NULL); */
    /* iftDestroyDataSet(&Z); */
    /* Z=Z_temp; */

    
    iftGraph *graph = iftCreateGraph(Z);
    graph->is_complete = true;

    sprintf(aux, "./data_result/%s", argv[4]);
    iftMakeDir(aux);
    
    printf("\n%s\n", argv[4]);

    iftRandomSeed(initial_seed);
    for (int i = 0; i < rep; i++) {
        sprintf(filename, "data_result/%s/IterativeOPFFMax_", argv[4]);
        strcat(filename, argv[4]);
        strcat(filename, "_");
        sprintf(num, "%d", i);
        strcat(filename, num);
        iftIterativeOPF(graph, num_clusters, max_iterations, 0, 0);
        iftWriteCSVDataSetClustering(graph->Z, filename);
        printf("Running experiments IterativeOPFFMax_%s %d of %d\n", argv[4], i + 1, rep);
        graph->centroids = NULL;
    }
    puts("");

    iftDestroyGraph(&graph);
    iftDestroyDataSet(&Z);

    Z = iftReadCSVDataSet(argv[1], ',', 1, NULL);
    Z->ntrainsamples = Z->nsamples;
    iftSetStatus(Z, IFT_TRAIN);
    graph = iftCreateGraph(Z);
    graph->is_complete = true;

    iftRandomSeed(initial_seed);
    for (int i = 0; i < rep; i++) {
        sprintf(filename, "data_result/%s/IterativeOPFFSum_", argv[4]);
        strcat(filename, argv[4]);
        strcat(filename, "_");
        sprintf(num, "%d", i);
        strcat(filename, num);
        iftIterativeOPF(graph, num_clusters, max_iterations, 1, 0);
        iftWriteCSVDataSetClustering(graph->Z, filename);
        printf("Running experiments IterativeOPFFSum_%s %d of %d\n", argv[4], i + 1, rep);
        graph->centroids = NULL;
    }
    puts("");

    iftDestroyGraph(&graph);
    iftDestroyDataSet(&Z);

    Z = iftReadCSVDataSet(argv[1], ',', 1, NULL);
    Z->ntrainsamples = Z->nsamples;
    iftSetStatus(Z, IFT_TRAIN);
    graph = iftCreateGraph(Z);
    graph->is_complete = true;

    iftRandomSeed(initial_seed);
    for (int i = 0; i < rep; i++) {
        sprintf(filename, "data_result/%s/IteratedWatershedsFMax_", argv[4]);
        strcat(filename, argv[4]);
        strcat(filename, "_");
        sprintf(num, "%d", i);
        strcat(filename, num);
        iftIteratedWatersheds(graph, num_clusters, max_iterations, IFT_MAX, IFT_DATA_FEAT, false);
        iftWriteCSVDataSetClustering(graph->Z, filename);
        printf("Running experiments IteratedWatershedsFMax_%s %d of %d\n", argv[4], i + 1, rep);
        graph->centroids = NULL;
    }
    puts("");

    iftDestroyGraph(&graph);
    iftDestroyDataSet(&Z);

    Z = iftReadCSVDataSet(argv[1], ',', 1, NULL);
    Z->ntrainsamples = Z->nsamples;
    iftSetStatus(Z, IFT_TRAIN);
    graph = iftCreateGraph(Z);
    graph->is_complete = true;

    iftRandomSeed(initial_seed);
    for (int i = 0; i < rep; i++) {
        sprintf(filename, "data_result/%s/IteratedWatershedsFSum_", argv[4]);
        strcat(filename, argv[4]);
        strcat(filename, "_");
        sprintf(num, "%d", i);
        strcat(filename, num);
        iftIteratedWatersheds(graph, num_clusters, max_iterations, IFT_SUM, IFT_DATA_FEAT, false);
        iftWriteCSVDataSetClustering(graph->Z, filename);
        printf("Running experiments IteratedWatershedsFSum_%s %d of %d\n", argv[4], i + 1, rep);
        graph->centroids = NULL;
    }
    puts("");

    iftDestroyGraph(&graph);
    iftDestroyDataSet(&Z);

    Z = iftReadCSVDataSet(argv[1], ',', 1, NULL);
    Z->ntrainsamples = Z->nsamples;
    iftSetStatus(Z, IFT_TRAIN);
    graph = iftCreateGraph(Z);
    graph->is_complete = true;

    iftRandomSeed(initial_seed);
    for (int i = 0; i < rep; i++) {
        sprintf(filename, "data_result/%s/KMeans_", argv[4]);
        strcat(filename, argv[4]);
        strcat(filename, "_");
        sprintf(num, "%d", i);
        strcat(filename, num);
        iftKMeans(graph->Z, num_clusters, max_iterations, 0.001, NULL, NULL, iftEuclideanDistance);
        iftWriteCSVDataSetClustering(graph->Z, filename);
        printf("Running experiments KMeans_%s %d of %d\n", argv[4], i + 1, rep);
        graph->centroids = NULL;
    }
    puts("");

    iftDestroyGraph(&graph);
    iftDestroyDataSet(&Z);

    /* Experiments with knn graphs */

    iftIntArray *knn_values = iftCreateIntArray(9);
    knn_values->val[0] = 5;
    knn_values->val[1] = 10;
    knn_values->val[2] = 15;
    knn_values->val[3] = 20;
    knn_values->val[4] = 30;
    knn_values->val[5] = 40;
    knn_values->val[6] = 50;
    knn_values->val[7] = 75;
    knn_values->val[8] = 100;

    for (int k = 0; k < knn_values->n; k++) {
      Z = iftReadCSVDataSet(argv[1], ',', 1, NULL);
        Z->ntrainsamples = Z->nsamples;
        iftSetStatus(Z, IFT_TRAIN);
        graph = iftCreateConnectedKnnGraph(Z, knn_values->val[k], false, num_clusters, NULL);

        iftRandomSeed(initial_seed);
        for (int i = 0; i < rep; i++) {
            sprintf(filename, "data_result/%s/IterativeOPFFMaxKNN%d_", argv[4], knn_values->val[k]);
            strcat(filename, argv[4]);
            strcat(filename, "_");
            sprintf(num, "%d", i);
            strcat(filename, num);
            iftIterativeOPF(graph, num_clusters, max_iterations, 0, 0);
            iftWriteCSVDataSetClustering(graph->Z, filename);
            printf("Running experiments IterativeOPFFMaxKNN%d_%s %d of %d\n", knn_values->val[k], argv[4], i + 1, rep);
            graph->centroids = NULL;
        }
        puts("");

        iftDestroyGraph(&graph);
        iftDestroyDataSet(&Z);

        Z = iftReadCSVDataSet(argv[1], ',', 1, NULL);
        Z->ntrainsamples = Z->nsamples;
        iftSetStatus(Z, IFT_TRAIN);
        graph = iftCreateConnectedKnnGraph(Z, knn_values->val[k], false, num_clusters, NULL);

        iftRandomSeed(initial_seed);
        for (int i = 0; i < rep; i++) {
            sprintf(filename, "data_result/%s/IterativeOPFFSumKNN%d_", argv[4], knn_values->val[k]);
            strcat(filename, argv[4]);
            strcat(filename, "_");
            sprintf(num, "%d", i);
            strcat(filename, num);
            iftIterativeOPF(graph, num_clusters, max_iterations, 1, 0);
            iftWriteCSVDataSetClustering(graph->Z, filename);
            printf("Running experiments IterativeOPFFSumKNN%d_%s %d of %d\n", knn_values->val[k], argv[4], i + 1, rep);
            graph->centroids = NULL;
        }
        puts("");

        iftDestroyGraph(&graph);
        iftDestroyDataSet(&Z);

        Z = iftReadCSVDataSet(argv[1], ',', 1, NULL);
        Z->ntrainsamples = Z->nsamples;
        iftSetStatus(Z, IFT_TRAIN);
        graph = iftCreateConnectedKnnGraph(Z, knn_values->val[k], false, num_clusters, NULL);

        iftRandomSeed(initial_seed);
        for (int i = 0; i < rep; i++) {
            sprintf(filename, "data_result/%s/IteratedWatershedsFMaxKNN%d_", argv[4], knn_values->val[k]);
            strcat(filename, argv[4]);
            strcat(filename, "_");
            sprintf(num, "%d", i);
            strcat(filename, num);
            iftIteratedWatersheds(graph, num_clusters, max_iterations, IFT_MAX, IFT_DATA_FEAT, false);
            iftWriteCSVDataSetClustering(graph->Z, filename);
            printf("Running experiments IteratedWatershedsFMaxKNN%d_%s %d of %d\n", knn_values->val[k], argv[4], i + 1, rep);
            graph->centroids = NULL;
        }
        puts("");

        iftDestroyGraph(&graph);
        iftDestroyDataSet(&Z);

        Z = iftReadCSVDataSet(argv[1], ',', 1, NULL);
        Z->ntrainsamples = Z->nsamples;
        iftSetStatus(Z, IFT_TRAIN);
        graph = iftCreateConnectedKnnGraph(Z, knn_values->val[k], false, num_clusters, NULL);

        iftRandomSeed(initial_seed);
        for (int i = 0; i < rep; i++) {
            sprintf(filename, "data_result/%s/IteratedWatershedsFSumKNN%d_", argv[4], knn_values->val[k]);
            strcat(filename, argv[4]);
            strcat(filename, "_");
            sprintf(num, "%d", i);
            strcat(filename, num);
            iftIteratedWatersheds(graph, num_clusters, max_iterations, IFT_SUM, IFT_DATA_FEAT, false);
            iftWriteCSVDataSetClustering(graph->Z, filename);
            printf("Running experiments IteratedWatershedsFSumKNN%d_%s %d of %d\n", knn_values->val[k], argv[4], i + 1, rep);
            graph->centroids = NULL;
        }
        puts("");

        iftDestroyGraph(&graph);
        iftDestroyDataSet(&Z);
    }

    return 0;
}
