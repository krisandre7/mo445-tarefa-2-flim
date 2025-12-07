//
// Created by peixinho on 29/11/16.
//

#include "ift.h"

double print(iftDict* problem, iftDict* params) {
    iftPrintDict(params);
    return 0.0;
}

double printClassifier(iftDict* problem, iftDataSet* Z, iftDict* params) {

    iftPrintDict(params);
    printf("%d %d %d\n", Z->nsamples, Z->nfeats, Z->ntrainsamples);

    return 0.0;
}

double printDescriptor(iftDict* problem, iftFileSet* Z, iftDict* params) {

    iftPrintDict(params);

    return 0.0;
}

int main() {


    return 0;
}