#ifndef DATASETVIEWER_H
#define DATASETVIEWER_H

#include "graphwidget.h"

class DataSetViewer : public GraphWidget
{
private:
    float padding;
    iftSampleStatus showStatus;
    iftDataSet* dataset;
    iftDataSet* reducedDataset;
    iftColorTable* classColorTable;

    Node* sampleToNode(iftSample& sample);
    void updateNodes();
public:
    DataSetViewer(QWidget *parent = 0);

    iftDataSet *getDataSet() const;
    void setDataSet(iftDataSet *value);

    iftDataSet* getReducedDataSet() const;

    iftColor classColor(int label);
    float getPadding() const;
};

#endif // DATASETVIEWER_H
