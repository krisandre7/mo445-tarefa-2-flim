#include "datasetviewer.h"
#include "node.h"
#include <QDebug>

DataSetViewer::DataSetViewer(QWidget* parent)
    :GraphWidget(parent)
{
    showStatus = ALL;
    classColorTable = NULL;
    reducedDataset = NULL;
    padding = 0.10;
}

void DataSetViewer::setDataSet(iftDataSet *value)
{
    dataset = value;

    if(reducedDataset!=NULL)
        iftDestroyDataSet(&reducedDataset);
    if(classColorTable!=NULL)
        iftDestroyColorTable(&classColorTable);

    if(dataset->projection == NULL) {
        qDebug() << dataset->nsamples << dataset->nfeats;
        reducedDataset = iftDimReductionByTSNE(dataset, 2, 40, 1000, false);
    }

//    int idx = -100;void DataSetViewer::resizeEvent(QResizeEvent *event)

//    float min = 100.0;
//    for (int i = 0; i < reducedDataset->nsamples; ++i) {
//        if(reducedDataset->sample[i].feat[0] < min) {
//            idx = i;
//            min = reducedDataset->sample[i].feat[0];
//        }
//    }

//    printf("%f\n", iftDistance1(dataset->sample[idx - 1].feat, dataset->sample[0].feat, dataset->alpha, dataset->nfeats));
//    printf("%f\n", iftDistance1(reducedDataset->sample[idx - 1].feat, reducedDataset->sample[0].feat, reducedDataset->alpha, reducedDataset->nfeats));

//    printf("%f\n", iftDistance1(dataset->sample[idx].feat, dataset->sample[0].feat, dataset->alpha, dataset->nfeats));
//    printf("%f\n", iftDistance1(reducedDataset->sample[idx].feat, reducedDataset->sample[0].feat, reducedDataset->alpha, reducedDataset->nfeats));


    iftPrintDataSetInfo(dataset, false);
    iftMinMaxFeatureScale(reducedDataset);
    iftPrintDataSetInfo(reducedDataset, false);
    fflush(stdout);
    classColorTable = iftCreateColorTable(dataset->nclasses, RGB_CSPACE);

    updateNodes();
}

iftDataSet *DataSetViewer::getReducedDataSet() const
{
    return reducedDataset;
}

iftColor DataSetViewer::classColor(int label)
{
    return classColorTable->color[label-1];
}

float DataSetViewer::getPadding() const
{
    return padding;
}

void DataSetViewer::updateNodes()
{
    clear();
    QString info("id: %1\n(%2, %3)\nclass: %4");
    for(int s = 0; s < reducedDataset->nsamples; ++s) {
        iftSample sample = reducedDataset->sample[s];
        printf(">>%d => %d\n", sample.id, sample.truelabel);
        if(sample.status & showStatus) {
            Node* node = sampleToNode(sample);
            node->setToolTip(info.arg(QVariant(sample.id).toString(),
                                      QVariant(sample.feat[0]).toString(),
                                      QVariant(sample.feat[1]).toString(),
                                      QVariant(sample.truelabel).toString()));
            addNode(node, sample.id);
        }
    }
}

Node *DataSetViewer::sampleToNode(iftSample &sample)
{

    QRectF rect = sceneRect();

    Node* node = new Node(this);
    float width = rect.width();
    float height = rect.height();

    float x = sample.feat[0];
    float y = sample.feat[1];

    qDebug() << sample.truelabel << sample.id;

    iftColor color = classColorTable->color[sample.truelabel-1];
    node->setColor(color.val[0], color.val[1], color.val[2]);
    node->setColorBorder(color.val[0]*0.4, color.val[1]*0.4, color.val[2]*0.4);

    qDebug() << x << y;

    x = x * (width - 2.0*width*padding) + width*padding + rect.left();
    y = y * (height - 2.0*height*padding) + height*padding + rect.top();

    qDebug() << x << y;


    qDebug() << "==============";

    QPointF pos(x, y);

    node->setPos(pos.x(), pos.y());
    node->setFlag(QGraphicsItem::ItemIsSelectable);

    sample.feat[0] = node->x();
    sample.feat[1] = node->y();

    return node;
}

iftDataSet *DataSetViewer::getDataSet() const
{
    return dataset;
}
