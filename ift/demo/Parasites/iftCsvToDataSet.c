//
// Created by peixinho-local on 15/05/17.
//

#include "ift.h"

iftDataSet* iftReadCelsoDataSet(const char* filepath) {
    iftCSV* csv = iftReadCSV(filepath, ',');

    iftDataSet* dataset = iftCreateDataSet(csv->nrows, csv->ncols - 1);

    for (int i=0; i < csv->nrows; ++i) {
        int f = 0;
        int id = iftImageSampleId(csv->data[i][0]);
        int truelabel = iftImageSampleLabel(csv->data[i][0]);

        dataset->sample[i].id = id;
        dataset->sample[i].truelabel = truelabel;

        for(int j=1; j<csv->ncols; ++j) {
            float val = 0.0f;
            sscanf(csv->data[i][j], "%f", &val);

            dataset->sample[i].feat[f] = val;
            f++;
        }
    }
    iftCountNumberOfClassesDataSet(dataset);

    iftDestroyCSV(&csv);

    return dataset;
}

int main(int argc, char** argv) {

    iftDataSet* Z = iftReadCelsoDataSet(argv[1]);

    iftPrintDataSetInfo(Z);

//    iftWriteOPFDataSet(Z, argv[2]);

    iftDestroyDataSet(&Z);

    return 0;

}
