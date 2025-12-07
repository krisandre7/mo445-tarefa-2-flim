#include "ift.h"

void iftLoadIris(double **data, int **label, int n, int d)
{
    *data = iftAlloc(n * d, sizeof **data);
    *label = iftAlloc(n, sizeof **label);

    FILE *f = fopen("iris-data.csv", "r");
    for (int i = 0; i < n*d; i+=4) {
        fscanf(f, "%lf,%lf,%lf,%lf\n", &(*data)[i], &(*data)[i+1], &(*data)[i+2], &(*data)[i+3]);
    }
    fclose(f);

    f = fopen("iris-label.csv", "r");
    for (int i = 0; i < n; i++) {
        fscanf(f, "%d ", &(*label)[i]);
    }
    fclose(f);
}

void iftPrintDouble(  double *data, int n, int d)
{
    printf("[");
    for (int i = 0; i < n; i++) {
        printf("[");
        for (int j = 0; j < d; j++) {
            printf("%lf", data[i * d + j]);
            if (j != (d-1)) printf(",");
        }
        printf("]");
        if (i != (n-1)) printf(",\n");
    }
    printf("]\n");
}

int main(int argc, const char *argv[])
{
    if (argc != 4) {
        iftError("iftIrisDemo <number of targets> <number of iterations (1000)> <learning rate (1e-7)>", "iftIrisDemo");
    }

    int k_targets = atoi(argv[1]);
    int iterations = atoi(argv[2]);
    double learn_rate = atof(argv[3]);

    int n = 150;
    int d = 4;
    int d_out = 2;
    double *data = NULL;
    int *label = NULL;
    iftLoadIris(&data, &label, n, d);

    iftCheckNumberOfTargets(label, n, &k_targets);
    double *new_L = iftLMCompAnalysis(data, label, NULL, n, d, d_out, k_targets, learn_rate, iterations, true);

    iftPrintDouble(new_L, d_out, d);

    iftFree(data);
    iftFree(label);
    iftFree(new_L);

    return 0;
}
