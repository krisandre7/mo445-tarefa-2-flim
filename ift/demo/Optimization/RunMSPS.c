//
// Created by peixinho on 6/23/15.
//

#include <ift.h>
#include <iftFunctions.h>

int DIMENSIONS = 100;
float D = 5.0;

float fitnessFunction(void* problem, float* x)
{
    return iftRastringinFunction(x, DIMENSIONS);
}

float initDelta(iftMSPS* msps, float min, float max, float d)
{
    for (int i = 0; i < msps->n; ++i) {
        for (int j = 0; j < msps->m; ++j) {
            int idx = iftGetMatrixIndex(msps->delta, i, j);
            float m = msps->m;
            msps->delta->val[idx] = (pow(j, d)/(2*pow(m, d)))*(max-min);
            msps->sigma->val[idx] = 1e-4 + ((j + 1.0) /(m + 1.0))*(1e-3 - 1e-4);
        }
        msps->min[i] = min;
        msps->max[i] = max;
    }
}

int main(int argc, char** argv)
{
    float min = -5;
    float max = 5;

    iftMSPS* msps = iftCreateMSPS(DIMENSIONS, 30, fitnessFunction, NULL);
    initDelta(msps, min, max, D);

    for (int i = 0; i < DIMENSIONS; ++i) {
        msps->theta[i] = min + (iftRandomInteger(0,100)/100.0)*(max - min);
    }

//    msps->iftPerturbation = iftMSPSLinearRandomPerturbation;

    printf("Running %d iterations ...\n", msps->niters);

//    msps->iftMinMaxPerturbation = iftMSPSDeterministicMinMaxPerturbation;
    float x = iftMSPSMin(msps);

    printf("f(x*) = %f at x* = ", x);

    printf("[%f", msps->theta[0]);
    for (int i = 0; i < DIMENSIONS; ++i) {
        printf(", %f", msps->theta[i]);
    }
    printf("]\n");

    printf("In %d iterations.\n", msps->iterstar);

    return 0;
}
