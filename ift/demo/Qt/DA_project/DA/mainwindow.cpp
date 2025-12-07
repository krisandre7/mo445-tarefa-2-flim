#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <datasetutils.h>

QMessageBox* MainWindow::messageBox = (QMessageBox*)NULL;

void iftErrorCatch(const char* func, const char* msg) {

    MainWindow::messageBox->critical(0, func, msg);

    qDebug() << "error on " << func << ": " << msg << '\n';

}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    MainWindow::messageBox = new QMessageBox(this);
    iftOnError = iftErrorCatch;
    ui->setupUi(this);
    initComponents();
}

void MainWindow::datasetLoaded () {
    clearAllData();
    originalDataset = DataSetUtils::instance()->getDataset();
//    noImpDataset = iftSelectNegativeSamples(originalDataset, originalDataset->nclasses);//remove samples from last class
    iftSetStatus(originalDataset, IFT_TRAIN);
    artificialDataset = iftCopyDataSet(originalDataset, true);

    float sum = 0.0;

    for (int i = 0; i < originalDataset->data->n; ++i) {
        sum += fabs(artificialDataset->data->val[i] - originalDataset->data->val[i]);
    }

    printf(">> %f\n", sum);

    iftCompareDataSets(originalDataset, artificialDataset, true, true);

    for (int s = 0; s < originalDataset->nsamples; ++s) {
        for (int f = 0; f < originalDataset->nfeats; ++f) {
            printf("[%d] %f", f, originalDataset->sample[s].feat[f]);
        }
        printf("\n");
        for (int f = 0; f < originalDataset->nfeats; ++f) {
            printf("[%d] %f", f, artificialDataset->sample[s].feat[f]);
        }
        printf("\n==\n");
    }


    printf("%p %p\n", originalDataset->data->val, artificialDataset->data->val);
    for (int s = 0; s < originalDataset->nsamples; ++s) {
        printf("%p %p\n", originalDataset->sample[s].feat, artificialDataset->sample[s].feat);
    }

    iftPrintDataSetInfo(originalDataset, false);

    kfold = iftKFold(originalDataset->nsamples, 5);

    fillSampleHash();
    if(applyPCA) {
        preProcessing = applyPCAOnDataset(originalDataset, pca_nComponents);
    }
    else if(applyLDA) {
        preProcessing = applyLDAOnDataset(originalDataset);
    }
    else {
        preProcessing = iftCopyMatrix(originalDataset->data);
    }
    fillSample();
    updateData();
}

iftMatrix* MainWindow::weightedAverageColumn(iftMatrix *matrix, float *weight)
{
    iftFloatArray* column = iftCreateFloatArray(matrix->nrows);
    iftMatrix* out = iftCreateMatrix(matrix->ncols, 1);

    for (int c = 0; c < matrix->ncols; ++c) {

        for (int r = 0; r < matrix->nrows; ++r) {
            column->val[r] = iftMatrixElem(matrix, c, r);
        }

        //printf("col = %d\n================\n", c);
        //iftPrintFloatArray(column->val, column->n);

        iftMatrixElem(out, c, 0) = iftWeightedMeanFloatArray(column->val, column->n, weight);
    }

    //printf("%p %p\n", column, column->val);
    //iftPrintFloatArray(column->val, column->n);
    fflush(stdout);
    iftDestroyFloatArray(&column);
    return out;
}

iftMatrix* MainWindow::weightedMedianColumn(iftMatrix *matrix, float *weight)
{
    iftFloatArray* column = iftCreateFloatArray(matrix->nrows);
    iftMatrix* out = iftCreateMatrix(matrix->ncols, 1);

    for (int c = 0; c < matrix->ncols; ++c) {

        for (int r = 0; r < matrix->nrows; ++r) {
            column->val[r] = iftMatrixElem(matrix, c, r);
        }

        //printf("col = %d\n================\n", c);
        //iftPrintFloatArray(column->val, column->n);
        float val = iftWeightedMedianFloatArray(column->val, column->n, weight);
        iftMatrixElem(out, c, 0) = val;
    }

    //printf("%p %p\n", column, column->val);
    //iftPrintFloatArray(column->val, column->n);
    fflush(stdout);
    iftDestroyFloatArray(&column);
    return out;
}

void MainWindow::initComponents(){
    setWindowTitle("IFT Data Augmentation");
    initMessageBox();
    initGraphicsViewProjectionArea();
    initGraphWidgets();
    initComponentsConnections();
    initLocalVariables();
    //TODO
    drawingManager.drawOption = COLOR;
    drawingManager.objectOption = NONE;

    kfold = NULL;

    //drawingManager.nodeColorOption = COLOR_NONE;
    //drawingManager.nodeHiddenOption = HIDDEN_NONE;
}

void MainWindow::initMessageBox(){
    MainWindow::messageBox->setFixedSize(500,200);
}

//this->formDisplaySampleImage = new FormDisplaySampleImage(this,hashIconIndex2FileLocation[index.row()]);
//this->formDisplaySampleImage->setAttribute( Qt::WA_DeleteOnClose );
//this->formDisplaySampleImage->show();

//area(component) where the scene will be placed in
void MainWindow::initGraphicsViewProjectionArea(){
    ui->graphicsViewProjectionArea->setStyleSheet("border: 1px solid black");
    ui->graphicsViewProjectionArea->setDragMode(QGraphicsView::RubberBandDrag);
    ui->graphicsViewProjectionArea->setInteractive(true);
}

//scene
void MainWindow::initGraphWidgets(){
    graphWidget = ui->graphicsViewProjectionArea;
    graphWidget->setDragMode(QGraphicsView::RubberBandDrag);
    graphWidget->scale(0.95, 0.95);
}

void MainWindow::initLocalVariables() {

    dialogUpdateSamplesImageDirectory = NULL;
    directory = QString("/images");
    prefix = QString("sample");
    changes = false;
    lastId = 100000;
    originalImageHeight = 28;
    originalImageWidth = 28;
    originalNchannels = 1;
    applyPCA = false;
    applyLDA = false;
    pca_nComponents = ui->pca_componentsDoubleSpinBox->value();
    isSampleAnImage = true;
    lastDialogPath = QDir::homePath();
}

void MainWindow::initComponentsConnections(){
    connect(graphWidget->scene(),SIGNAL(selectionChanged())
            ,this,SLOT(updateSelectedSamples()));

    connect(graphWidget,SIGNAL(mouseMoveCoordinates(QPointF* ,QPointF*,bool)),
            this,SLOT(mouseMoveScene(QPointF*,QPointF*, )));

    connect(graphWidget,SIGNAL(mousePressed(QPointF* ,QPointF*, Qt::MouseButton)),
            this,SLOT(mouseClicked(QPointF*, QPointF*, Qt::MouseButton)));

    connect(graphWidget,SIGNAL(deleteObjectSignals()),
            this,SLOT(removeNodes() ));

    connect(DataSetUtils::instance(), SIGNAL(loadFinished()),
            this, SLOT(datasetLoaded()));

}

iftSample* MainWindow::interpolateFeatures(iftSample newPoint)
{
    iftDataSet* reducedDataset = graphWidget->getReducedDataSet();
    iftDataSet* dataset = originalDataset;

    float *distances = iftAllocFloatArray(reducedDataset->nsamples);
    int* index = iftAllocIntArray(reducedDataset->nsamples);

    for (int i = 0; i < reducedDataset->nsamples; ++i) {
        iftSample sample = reducedDataset->sample[i];
        distances[i] = iftDistance1(sample.feat, newPoint.feat,
                                    reducedDataset->alpha, reducedDataset->nfeats);
        index[i] = i;
    }

    float* distancesold = iftAllocFloatArray(reducedDataset->nsamples);
    iftCopyFloatArray(distancesold, distances, reducedDataset->nsamples);

    iftPrintFloatArray(distances, reducedDataset->nsamples);

    iftFHeapSort(distances, index, reducedDataset->nsamples, IFT_INCREASING);

    printf("\n\n==========================\n");
    for (int i = 0; i < reducedDataset->nsamples; ++i) {
        printf("%f ", distancesold[index[i]]);
    }
    printf("\n");
    fflush(stdout);

    int k = knearest;

    printf("\n\n========================\n(%f, %f)\n", newPoint.feat[0], newPoint.feat[1]);
    for (int i = 0; i < k; ++i) {
        printf("%d => %d %d %f (%f, %f)\n", i, index[i], reducedDataset->sample[index[i]].truelabel, distancesold[index[i]], reducedDataset->sample[index[i]].feat[0], reducedDataset->sample[index[i]].feat[1]);
    }
    printf("\n"); fflush(stdout);

    printf("\n\n========================\n(%f, %f)\n", newPoint.feat[0], newPoint.feat[1]);
    for (int i = 0; i < k; ++i) {
        printf("%d %d %d %d\n", reducedDataset->sample[index[i]].truelabel, dataset->sample[index[i]].truelabel, reducedDataset->sample[index[i]].id, dataset->sample[index[i]].id);
    }
    printf("\n"); fflush(stdout);

    printf("\n============= (%d,%d)\n", dataset->nsamples, reducedDataset->nsamples);
    for (int s = 0; s < dataset->nsamples; ++s) {
        printf(">> %d %d\n", dataset->sample[s].id, reducedDataset->sample[s].id);
    }

    iftMatrix* feats = iftCreateMatrix(dataset->nfeats, k);

    for(int i = 0; i < k; ++i) {
//        Edge* edge = new Edge(graphWidget->node(index[i]), node);
//        graphWidget->addEdge(edge);
//        distances[i] = (1 + 1e-5 ) / (1e-5 + distances[i]);
        for (int f = 0; f < dataset->nfeats; ++f) {
            iftMatrixElem(feats, f, i) = dataset->sample[index[i]].feat[f];
        }
    }

    iftIntArray* classes = iftCreateIntArray(dataset->nclasses + 1);

    for (int i = 0; i < k; ++i) {
        printf("%d\n", reducedDataset->sample[index[i]].truelabel);
        classes->val[reducedDataset->sample[index[i]].truelabel]++;
    }

    printf("classes = ");
    iftPrintIntArray(classes->val, classes->n);
    printf("\n");

    int truelabel = iftArgmax(classes->val, classes->n);
    iftDestroyIntArray(&classes);

    iftFloatArray* weight = iftCreateFloatArray(k);

    for (int i = 0; i < weight->n; ++i) {
        weight->val[i] = 1.00001/(1e-5 + distances[index[i]]);
//                weight->val[i] = 1.0;
    }

    iftMatrix* interpolated = weightedMedianColumn(feats, weight->val);

    //    iftPrintMatrix(feats);
    //    printf("\n===================\n");
    //    iftPrintMatrix(interpolated);
    //    printf("\n===================\n");


    iftFree(distances);
    iftFree(index);
    iftDestroyMatrix(&feats);

    iftSample* newSample = (iftSample*) iftAlloc(1, sizeof(iftSample));
    newSample->feat = iftAllocFloatArray(dataset->nfeats);
    newSample->truelabel = truelabel;

    iftCopyFloatArray(newSample->feat, interpolated->val, interpolated->ncols);
    iftDestroyMatrix(&interpolated);

    //iftPrintFloatArray(newSample->feat, dataset->nfeats);

    return newSample;
}

void MainWindow::removeNodes(){
    if(graphWidget->nodes().size() <= 0 ){
        return;
    }
    int index = 0;

    QList<QGraphicsItem*> selected = graphWidget->scene()->selectedItems();
    qDebug() << selected.size();
    for (QGraphicsItem* item : selected) {
        Node* node = (Node*) item;
        int sampleId = hashNodeItemPointer2sampleId.value(node);
        int index = hashNodeItemPointer2GraphNodeIndex.value(node);
        qDebug() << "Index: " << index;
        qDebug() << "Id: " << sampleId;
        artificialDataset->sample[index].status = IFT_OUTLIER;
        graphWidget->removeNode(index);
        delete node;
    }

    graphWidget->update();
}


void MainWindow::fillSampleHash(){
    for (int sampleIndex = 0; sampleIndex < originalDataset->nsamples; ++sampleIndex) {
        iftSample* sample = (iftSample*)calloc(1,sizeof(iftSample));
        sample->feat = iftAllocFloatArray(originalDataset->nfeats);
        qDebug() << sampleIndex;
        iftCopySample(&(originalDataset->sample[sampleIndex]),sample,originalDataset->nfeats,true);
        listSample.append(sample);
        listSampleKused.append(0);
        hashSampleId2iftSample.insert(sample->id,sample);
        hashSampleId2Kused.insert(sample->id,0);

    }
}

void MainWindow::fillSample(){
    datasetPreProcessing = iftFeatureMatrixToDataSet(preProcessing);
    datasetPreProcessing->nclasses = originalDataset->nclasses;

    int nprojection = 0;

    if(originalDataset->projection) {
        datasetPreProcessing->projection = iftCopyDoubleMatrix(originalDataset->projection);
        nprojection = datasetPreProcessing->projection->ncols;
    }
    for (int sampleIdx = 0; sampleIdx < datasetPreProcessing->nsamples; ++sampleIdx) {
        datasetPreProcessing->sample[sampleIdx].truelabel =  originalDataset->sample[sampleIdx].truelabel;
        datasetPreProcessing->sample[sampleIdx].id =  originalDataset->sample[sampleIdx].id;
        datasetPreProcessing->sample[sampleIdx].group =  originalDataset->sample[sampleIdx].group;
        if(datasetPreProcessing->projection)
            datasetPreProcessing->sample[sampleIdx].projection = &(datasetPreProcessing->projection->val[nprojection*sampleIdx]);
        iftSample* sample = (iftSample*)calloc(1,sizeof(iftSample));
        sample->feat = iftAllocFloatArray(datasetPreProcessing->nfeats);
        iftCopySample(&(datasetPreProcessing->sample[sampleIdx]),sample,datasetPreProcessing->nfeats,true);
        hashSampleId2iftSamplePCA.insert(sample->id,sample);
    }
}


void MainWindow::mouseClicked(QPointF* currentMousePositionInGraphicArea,
                              QPointF* currentMousePositionInScene, Qt::MouseButton buttons){

    if(!(buttons & Qt::LeftButton))
        return;

    mouseCoordInGraphicArea.setX(currentMousePositionInGraphicArea->x());
    mouseCoordInGraphicArea.setY(currentMousePositionInGraphicArea->y());
    mouseCoordInScene.setX(currentMousePositionInScene->x());
    mouseCoordInScene.setY(currentMousePositionInScene->y());

    if(graphWidget->nodes().size() > 0){
        if(graphWidget->itemAt(mouseCoordInGraphicArea.x(), mouseCoordInGraphicArea.y())){
            return;
        }
        iftDataSet* dataset = NULL;
        dataset = originalDataset;

        //        int size = dataset->nsamples;
        //        int k = knearest;
        //        int* distances = iftAllocIntArray(size);
        //        int* indices = iftAllocIntArray(size);
        //        for (int var = 0; var < size; ++var) {
        //            Node* node = graphWidget->node(var);
        //            int deltaX = node->x() - mouseCoordInScene.x();
        //            int deltaY = node->y() - mouseCoordInScene.y();
        //            int distance = deltaX*deltaX + deltaY*deltaY;
        //            distances[var] = distance;
        //            indices[var] = var;
        //            for (int l = var; l > 0; --l) {
        //                if(distances[l-1] > distances[l]){
        //                    iftSwap(distances[l-1],distances[l]);
        //                    iftSwap(indices[l-1],indices[l]);
        //                }else{
        //                    break;
        //                }
        //            }
        //        }
        //        free(distances);
        //        iftMatrix* dataHighDimension = iftCreateMatrix(dataset->nfeats,k);
        //        iftMatrix*dataLowDimensionExpanded = iftCreateMatrix(6,k);

        //        int *countLabels = iftAllocIntArray(dataset->nclasses+1);
        //        int x,y;
        //        for (int nearestindex = 0; nearestindex < k; ++nearestindex) {

        //            for (int col = 0; col < dataHighDimension->ncols; ++col) {
        //                iftMatrixElem(dataHighDimension,col,nearestindex) = dataset->sample[indices[nearestindex]].feat[col];
        //            }
        //            Node* node =  graphWidget->node(indices[nearestindex]);
        //            x = node->x(); y = node->y();
        //            iftMatrixElem(dataLowDimensionExpanded,0,nearestindex) = 1;
        //            iftMatrixElem(dataLowDimensionExpanded,1,nearestindex) = x;
        //            iftMatrixElem(dataLowDimensionExpanded,2,nearestindex) = y;
        //            iftMatrixElem(dataLowDimensionExpanded,3,nearestindex) = x*x;
        //            iftMatrixElem(dataLowDimensionExpanded,4,nearestindex) = x*y;
        //            iftMatrixElem(dataLowDimensionExpanded,5,nearestindex) = y*y;

        //            countLabels[ dataset->sample[indices[nearestindex]].truelabel]++;
        //        }
        //        int indexMaxCount = iftArgmax(countLabels,dataset->nclasses+1);

        //        //free(dataLowDimension);
        //        iftMatrix* clickedPoint = iftCreateMatrix(2,1);
        //        iftMatrixElem(clickedPoint,0,0) =mouseCoordInScene.x();
        //        iftMatrixElem(clickedPoint,1,0) =mouseCoordInScene.y();
        //        iftMatrix* clickedPointExpanded = iftPolynomialExpasion(clickedPoint,2);
        //        iftMatrix* coefficients = iftLeastSquares(dataLowDimensionExpanded,dataHighDimension);
        //        iftMatrix* newPoint =iftMultMatrices(clickedPointExpanded,coefficients);

        //        iftMatrix* pointOriginalSpace;
        //        if(applyPCA){

        //            iftSample* newSamplePCA = (iftSample*)calloc(1,sizeof(iftSample));
        //            newSamplePCA->feat = iftAllocFloatArray(newPoint->n);
        //            newSamplePCA->truelabel = indexMaxCount;
        //            newSamplePCA->id = lastId;
        //            for (size_t var = 0; var < newPoint->n; ++var) {
        //                newSamplePCA->feat[var] = newPoint->val[var];
        //            }
        //            listSamplePCA.append(newSamplePCA);
        //            hashSampleId2iftSamplePCA.insert(newSamplePCA->id,newSamplePCA);
        //            iftMatrix* pointPCAWithPadding= iftCreateMatrix(datasetPreProcessing->nfeats,1);
        //            for (int index = 0; index < newPoint->ncols; ++index) {
        //                pointPCAWithPadding->val[index] = newPoint->val[index];
        //            }
        //            //qDebug() << pointPCAWithPadding->nrows << pointPCAWithPadding->ncols << Vt->nrows << Vt->ncols;
        //            pointOriginalSpace = iftMultMatrices(pointPCAWithPadding, Vt);
        //            iftDestroyMatrix(&pointPCAWithPadding);
        //        }else{
        //            pointOriginalSpace = newPoint;
        //        }
        //        iftSample* newSample = (iftSample*)calloc(1,sizeof(iftSample));
        //        newSample->feat = iftAllocFloatArray(pointOriginalSpace->n);
        //        newSample->truelabel = indexMaxCount;
        //        newSample->id = lastId;
        //        for (size_t var = 0; var < pointOriginalSpace->n; ++var) {
        //            newSample->feat[var] = pointOriginalSpace->val[var];
        //        }


        float width = graphWidget->frameSize().width();
        float height= graphWidget->frameSize().height();

        float padding = graphWidget->getPadding();

        iftSample s;
        s.feat = iftAllocFloatArray(2);
        s.feat[0] = mouseCoordInScene.x();
        s.feat[1] = mouseCoordInScene.y();

        s.truelabel = 1;

        Node *node = new Node(graphWidget);
        node->setPos(mouseCoordInScene);

//        int x = mouseCoordInScene.x();
//        int y = mouseCoordInScene.y();
//        node->setPos(qreal(x),qreal(y));
        node->setFlag(QGraphicsItem::ItemIsSelectable);
        node->setNodeShape(Rectangle);

        this->node = node;

        iftSample* newSample = interpolateFeatures(s);
        newSample->id = lastId++;
        newSample->status = IFT_ARTIFICIAL | IFT_TRAIN;

        iftAddSampleDataset(artificialDataset, *newSample);

        //hashSampleId2iftSample.insert(lastId,newSample);

        QImage image;
        QImage imageResized;
        //listSample.append(newSample);
        if(isSampleAnImage){

            if(originalNchannels == 1){

                uchar* data = (uchar*)calloc(dataset->nfeats, sizeof(uchar));
                for (size_t var = 0; var < dataset->nfeats; ++var) {
                    if(newSample->feat[var] < 0){
                        newSample->feat[var] = 0;
                    }
                    if(newSample->feat[var] > 255){
                        newSample->feat[var] = 255;
                    }
                    data[var] = newSample->feat[var];
                }
                image = QImage(data,originalImageHeight, originalImageWidth, QImage::Format_Grayscale8);
                imageResized = image.scaled(QSize(50,50));
            }
            else if(originalNchannels == 3){
                //QImage::Format_ARGB32
                uchar* data = (uchar*)calloc(dataset->nfeats,4*sizeof(uchar));
                int k = 0;
                for (size_t var = 0; var < dataset->nfeats; var += 3) {
                    data[k] = 255;
                    k++;
                    for (int c = 0; c < originalNchannels; ++c) {
                        if(newSample->feat[var+c] < 0){
                            newSample->feat[var+c] = 0;
                        }
                        if(newSample->feat[var+c] > 255){
                            newSample->feat[var+c] = 255;
                        }
                        data[k] = newSample->feat[var];
                        k++;
                    }
                }
                image = QImage(data,originalImageHeight, originalImageWidth, QImage::Format_ARGB32);
                imageResized = image.scaled(QSize(50,50));
            }
        }else{
            //TODO
        }
        //listSample.append(newSample);
        hashSampleId2iftSample.insert(newSample->id,newSample);


        char buff[100];
        if(!sprintf(buff,"%06d_%08d.png", newSample->truelabel, lastId)){
            iftError("lascou-c","mouseClicked");
        }
        image.save("3_1.png");
        QListWidgetItem *itemWidget = new QListWidgetItem(QIcon(QPixmap::fromImage(imageResized)),buff);
        itemWidget->setFlags(itemWidget->flags() & Qt::NoItemFlags);
        itemWidget->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEditable | Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
        hashsampleId2FileLocation.insert(currentNumberSamples,"createdImage");
        hashIconIndex2FileLocation.insert(currentNumberSamples,"createdImage");
        hashSampleId2ListWidgetItem.insert(newSample->id,itemWidget);

        int nnodes = graphWidget->nodes().size();

        hashNodeItemPointer2sampleId.insert(node,lastId);
        hashNodeItemPointer2GraphNodeIndex.insert(node, nnodes);

        graphWidget->addNode(node, lastId);
        iftColor color = graphWidget->classColor(newSample->truelabel);
        node->setColor(color.val[0],color.val[1],color.val[2]);
        node->setColorBorder(color.val[0]*0.4,color.val[1]*0.4, color.val[2]*0.4);
        //listSampleKused.append(k);
        //        hashSampleId2Kused.insert(newSample->id,k);
    }

}

void MainWindow::mouseMoveScene(QPointF* currentMousePositionInGraphicArea,
                                QPointF* currentMousePositionInScene){

    mouseCoordInGraphicArea.setX(currentMousePositionInGraphicArea->x());
    mouseCoordInGraphicArea.setY(currentMousePositionInGraphicArea->y());
    mouseCoordInScene.setX(currentMousePositionInScene->x());
    mouseCoordInScene.setY(currentMousePositionInScene->y());


    if(graphWidget->nodes().size() > 0){
        if(QGraphicsItem *item = graphWidget->itemAt(mouseCoordInGraphicArea.x(),mouseCoordInGraphicArea.y())){
            //int nodeIndex = hashNodeItemPointer2GraphNodeIndex[item];
            Node* node = qgraphicsitem_cast<Node *>(item);

            int sampleId = hashNodeItemPointer2sampleId[node];

            QListWidgetItem* itemWidget = hashSampleId2ListWidgetItem[sampleId];

            iftSample* sample = hashSampleId2iftSample[sampleId];

            int k = hashSampleId2Kused[sampleId];
            //ui->trueLabelComboBox->setCurrentIndex(sample->truelabel);
            //ui->k_usedSpinBox->setValue(k);
        }
    }
}


MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::clearAllData(){
    clearAllHashTable();
    clearAllDatasets();
    clearAllGraphNodes();
    clearAllGraphEdges();
}

void MainWindow::clearAllHashTable(){

    hashLabelId2LabelName.clear();
    hashNodeItemPointer2GraphNodeIndex.clear();
    hashsampleId2FileLocation.clear();
    hashsampleId2IconIndex.clear();
    hashsampleId2labelId.clear();
    hashsampleId2SupervisedSample.clear();
}

void MainWindow::clearAllDatasets(){
    if(kfold!=NULL)
        iftDestroySampler(&kfold);
    iftDestroyDataSet(&originalDataset);
    iftDestroyDataSet(&datasetPreProcessing);
}

void MainWindow::clearAllGraphNodes(){
    graphWidget->clear();
}

void MainWindow::clearAllGraphEdges(){
    graphKnnMap.clear();
    graphWidget->clear();
    graphWidget->update();
}

bool MainWindow::updateDataSetInfo(QString pathname){
    try{
        if(QFileInfo(pathname).suffix() == "zip") {
            //DataSetUtils::instance();
            DataSetUtils::instance()->load(pathname);
            setWindowTitle(pathname);
        }
    }
    catch (int e){
        messageBox->critical(0,"Error","File is not a iftdataset");
        return false;
    }

    return true;
}

iftMatrix* MainWindow::applyLDAOnDataset(iftDataSet* dataset) {

    iftSelectSupTrainSamples(dataset, 0.75);
//    dataset = iftAlignDataSetByPCA(dataset);
    iftMatrix* lda = iftLinearDiscriminantAnalysis(dataset);
    iftSetStatus(dataset, IFT_TRAIN);

    printf("====================\n");
    iftPrintFloatArray(dataset->sample[0].feat, dataset->nfeats);
    printf("====================\n");

    iftPrintMatrix(lda);
    fflush(stdout);

    iftDataSet* ldaDataset = iftDataSetTransformBasis(dataset, lda);
    iftDestroyMatrix(&lda);
    iftMatrix* output = iftDataSetToFeatureMatrix(ldaDataset);
    iftDestroyDataSet(&ldaDataset);

    return output;
}

iftMatrix* MainWindow::applyPCAOnDataset(iftDataSet* dataset, double nComponents){

    iftMatrix* S = NULL;
    iftSetStatus(dataset, IFT_TRAIN);
    iftDataSet* centralized = iftCentralizeDataSet(dataset);
    iftMatrix* pca = iftRotationMatrixAndSingularValuesByPCA(centralized, &S);
    iftDestroyDataSet(&centralized);

    iftUnitNorm(S->val, S->n);

    int ncols = 0;
    if(nComponents <= 1.0) {
        double sum = 0.0;
        for (int i = 0; i < S->n; ++i) {
            ncols++;
            sum += S->val[i];
            if(sum>=nComponents) {
                break;
            }
        }
    }
    else {
        ncols = iftMin(nComponents, pca->ncols);
    }

    iftMatrix* sub = iftSubMatrix(pca, 0, pca->nrows, 0, ncols);
    iftDestroyMatrix(&pca);
    iftDataSet* pcaDataset = iftDataSetTransformBasis(dataset, sub);

    Vt = iftTransposeMatrix(sub);
    iftDestroyMatrix(&sub);

    iftMatrix* output = iftDataSetToFeatureMatrix(pcaDataset);
    iftDestroyDataSet(&pcaDataset);

    return output;
}


void MainWindow::creatIconsOnThumb(){
    for (int i = 0; i < originalDataset->nsamples; ++i) {
        hashsampleId2IconIndex.insert(originalDataset->sample[i].id,i);
        QString sampleName = mountSampleName(&originalDataset->sample[i]);
        QListWidgetItem *itemWidget = new QListWidgetItem(QIcon("icons/imageMissing.png"),sampleName);
        itemWidget->setFlags(itemWidget->flags() & Qt::NoItemFlags);
        itemWidget->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEditable | Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
        hashsampleId2FileLocation.insert(originalDataset->sample[i].id,"icons/imageMissing.png");
        hashIconIndex2FileLocation.insert(originalDataset->sample[i].id,"icons/imageMissing.png");
        hashSampleId2ListWidgetItem.insert(originalDataset->sample[i].id,itemWidget);
    }
}


void MainWindow::openDataSet(){

    QString fileName_dataset = QFileDialog::getOpenFileName(this, tr("Open File"), lastDialogPath,
                                                            tr("Dataset Files (*.zip *.csv);;"));
    if (!fileName_dataset.isEmpty()) {
        if(updateDataSetInfo(fileName_dataset.toLatin1().data())) {

        }
    }else{
        messageBox->critical(0,"Error","File is empty");
    }
}

void MainWindow::updateData() {


    creatIconsOnThumb();

    //    datasetLowDimension = datasetDimensionalityReduction(datasetPreProcessing);
    //    for (int var = 0; var < datasetPreProcessing->nsamples; ++var) {
    //        qDebug() << datasetPreProcessing->sample[var].id << datasetLowDimension->sample[var].id;
    //    }
    //datasetLowDimension = datasetDimensionalityReduction(originalDataset);

    graphWidget->setDataSet(datasetPreProcessing);

    for (int i = 0; i < datasetPreProcessing->nsamples; ++i) {
        hashNodeItemPointer2sampleId.insert(graphWidget->node(i), datasetPreProcessing->sample[i].id);
        hashNodeItemPointer2GraphNodeIndex.insert(graphWidget->node(i), i);
    }

    knearest = 10;
    updateKNearestSpinBox();
}


void MainWindow::updateKNearestSpinBox(){
    ui->kNearestSpinBox->setValue(knearest);
    ui->kNearestSpinBox->setMaximum(originalDataset->nsamples);
}

void MainWindow::updateKnearest(){
    knearest = ui->kNearestSpinBox->value();
}

void MainWindow::updatePCA_nComponents(){
    pca_nComponents = ui->pca_componentsDoubleSpinBox->value();
}

void MainWindow::resizeEvent(QResizeEvent* event)
{
//        mainWindowHeight = event->size().height();
//        mainWindowWidth = event->size().width();
//        //int minDimension = iftMin(mainWindowHeight,mainWindowWidth);
//        sceneProjectionAreaHeight = ui->graphicsViewProjectionArea->height();
//        sceneProjectionAreaWidth = ui->graphicsViewProjectionArea->width();
//        sceneProjectionAreaX = ui->graphicsViewProjectionArea->width()*0.025;
//        sceneProjectionAreaY = ui->graphicsViewProjectionArea->width()*0.025;

//        graphWidget->setSceneRect(0,
//                                  0,
//                                  sceneProjectionAreaWidth,
//                                  sceneProjectionAreaHeight);

////        graphWidget->setMinimumSize(ui->graphicsViewProjectionArea->size());

//        graphWidget->scene()->update();
}


void MainWindow::createWindowToSetDirectory() {

    //is the window already displayed?
    if(dialogUpdateSamplesImageDirectory == NULL){//NO
        //create a window
        dialogUpdateSamplesImageDirectory = new DialogUpdateSamplesImageDirectory(this,prefix,directory);
        dialogUpdateSamplesImageDirectory->setAttribute( Qt::WA_DeleteOnClose);
        dialogUpdateSamplesImageDirectory->show();

        connect(dialogUpdateSamplesImageDirectory,SIGNAL(updateDirectoryAndPrefix(QString,QString)),
                this, SLOT(updatePrefixAndDirectory(QString,QString)));

        connect(dialogUpdateSamplesImageDirectory,SIGNAL(windowClosed()),this,SLOT(windowPrefixDirClosed()));
    }else{//YES
        dialogUpdateSamplesImageDirectory->activateWindow();
    }
}

void MainWindow::updatePrefixAndDirectory(QString prefix,QString directory){
    this->prefix = QString(prefix);
    this->directory = QString(directory);
    changes = true;
}


void MainWindow::windowPrefixDirClosed(){
    delete this->dialogUpdateSamplesImageDirectory;
    dialogUpdateSamplesImageDirectory = NULL;
    if (changes) {
        if(originalDataset != NULL){
            updateIconsOnThumb(originalDataset,true);
        }
    }
}

void MainWindow::updateIconsOnThumb(iftDataSet *dataset, bool showProgressBar){
    //finding the images file
    QDir dir = (directory);
    dir.setFilter(QDir::Files);
    dir.setSorting(QDir::Name);
    QFileInfoList list = dir.entryInfoList();

    if(showProgressBar){
        QProgressDialog progressDialog("Processing...", NULL,0, INT_MAX, this);
        QProgressBar* bar = new QProgressBar(&progressDialog);
        QLabel label;
        //progressDialog.setCancelButton(NULL);
        bar->setRange(0, 100);
        bar->setValue(0);
        progressDialog.setBar(bar);
        progressDialog.setMinimumWidth(350);
        progressDialog.setMinimumDuration(1000);
        progressDialog.setWindowModality(Qt::WindowModal);
        progressDialog.setValue(0);
        int percentage = 0;
        for (int i = 0; i < dataset->nsamples; ++i) {
            QString sampleName = mountSampleName(&dataset->sample[i]);
            QString absolutePath = directory + "/" + sampleName;

            //findSampleFile(list,sampleName,dataset->sample[i].id);
            updateSampleIcon(absolutePath,dataset->sample[i].id);
            percentage = ((float)i/list.size())*100;
            progressDialog.setValue(percentage);
        }
    }

}

QString MainWindow::mountSampleName(iftSample* sample){
    char name[100];
    sprintf(name,"%06d_%08d.png",sample->truelabel,sample->id);
    QString sampleName = QString(name);
    return sampleName;
}

void MainWindow::updateSampleIcon(QString imageFileAbsolutePath, int sampleId){
    QImage image(imageFileAbsolutePath);
    if(image.isNull()){
        printf("could not open file %s\n",imageFileAbsolutePath.toLatin1().data());
        return;
    }
    originalImageHeight = image.height();
    originalImageWidth = image.width();
    originalNchannels = image.isGrayscale() ? 1 : 3;
    QImage imageResized = image.scaled(QSize(50,50));

    int iconIndex = hashsampleId2IconIndex[sampleId];
    hashsampleId2FileLocation[sampleId] = imageFileAbsolutePath;
    hashIconIndex2FileLocation[iconIndex] = imageFileAbsolutePath;
    listPaths2Images.append(imageFileAbsolutePath);
}

void MainWindow::loadCategories(){
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"), lastDialogPath,
                                                    tr("All files (*)"));
    if (!fileName.isEmpty()) {
        lastDialogPath =  QFileInfo(fileName).absoluteDir().path();
        updateHashLabelName(fileName);
        messageBox->information(0,"Load Labels","Labels were loaded");
    }
}

bool MainWindow::updateHashLabelName(QString fileName_categories){
    QFile inputFile(fileName_categories);
    bool sucess = false;
    if (inputFile.open(QIODevice::ReadOnly))
    {
        QTextStream in(&inputFile);
        int i=0;
        hashLabelId2LabelName.clear();
        hashLabelId2LabelName.insert(i,"unknown");
        i++;

        while (!in.atEnd())
        {
            QString line = in.readLine();
            hashLabelId2LabelName.insert(i,line);
            i++;
        }
        inputFile.close();
        sucess = true;
    }
    else{
        sprintf(erroMessagesBuffer,"[updateHashLabelName] Could not open file %s", fileName_categories.toLatin1().data());
        messageBox->critical(0,"Error",erroMessagesBuffer);
        sucess = false;
    }
    return sucess;
}

void MainWindow::on_actionDataSet_triggered()
{
    openDataSet();
}

void MainWindow::validate() {

}

void MainWindow::on_spinBox_valueChanged(int arg1)
{
    arg1 += arg1 + 1;
    updateKnearest();
}

void MainWindow::on_pca_componentsDoubleSpinBox_valueChanged(double arg1)
{
    arg1 += arg1 + 1;
    updatePCA_nComponents();
}

void MainWindow::on_kNearestSpinBox_valueChanged(int arg1)
{
    arg1 += arg1 + 1;
    updateKnearest();
}


void MainWindow::changeSampleTrueLabel(int newTrueLabel){
    for (int var = 0; var < graphWidget->nodes().size(); ++var) {
        Node* node = graphWidget->node(var);
        if(node->isSelected()){
            iftColor color = graphWidget->classColor(newTrueLabel);
            node->setColor(color.val[0],color.val[1],color.val[2]);
            node->setColorBorder(color.val[0]*0.4,color.val[1]*0.4,color.val[2]*0.4);
            int sampleId = hashNodeItemPointer2sampleId[node];
            iftSample* sample = hashSampleId2iftSample[sampleId];
            sample->truelabel = newTrueLabel;

            if(applyPCA){
                iftSample* samplePCA = hashSampleId2iftSamplePCA[sampleId];
                samplePCA->truelabel = newTrueLabel;
            }
            node->update();
        }
    }
}

void MainWindow::on_pushButton_2_clicked()
{
    trainClassifier();
}

void MainWindow::trainClassifier() {

    iftDataSet *tempDataset = iftCopyDataSet(noImpDataset, false);
//    iftSetStatus(testDataset,IFT_TEST);

    ift_kernel_type kernel_type = IFT_RBF;
    ift_multi_class multiclass = IFT_OVA;
    float C = 1e4;
    float sigma = 0.10;

    iftSVM *svm = iftCreateSVM(kernel_type, multiclass, C, sigma);
    iftSetStatus(noImpDataset, IFT_TRAIN);
    iftSVMTrain(svm, noImpDataset);

    for (int m = 0; m < svm->nmodels; ++m) {
        printf("Model %d =========\n", m);
        for(int sv=0; sv < svm->model[m]->l; sv++) {
            int idx = svm->model[m]->sv_indices[sv];
            printf("SV: %d %d\n", idx, noImpDataset->sample[idx].truelabel);
            graphWidget->node(idx)->mark();
        }
    }

    graphWidget->scene()->update();
    graphWidget->viewport()->update();
}

void MainWindow::on_actionDataset_Test_triggered()
{
    QString fileName_dataset = QFileDialog::getOpenFileName(this, tr("Open File"), QString(),
                                                            tr("Dataset Files (*.zip *.csv);;"));
    if (!fileName_dataset.isEmpty()) {
        testDataset = iftReadOPFDataSet(fileName_dataset.toLatin1().data());
    }
}

void MainWindow::on_actionDataset_Validation_triggered()
{
    QString fileName_dataset = QFileDialog::getOpenFileName(this, tr("Open File"), QString(),
                                                            tr("Dataset Files (*.zip *.csv);;"));
    if (!fileName_dataset.isEmpty()) {
        validationDataset = iftReadOPFDataSet(fileName_dataset.toLatin1().data());
    }
}

void MainWindow::on_pushButton_3_clicked()
{
    saveDataset();
}

void MainWindow::saveDataset() {
    if(originalDataset){

        QString filename = QFileDialog::getSaveFileName(this, tr("Open File"), QString(),
                                                                tr("Dataset Files (*.zip);;"));

        if(filename.isEmpty()) {
            return;
        }

        iftWriteOPFDataSet(graphWidget->getReducedDataSet(), "/home/peixinho/analgesico.zip");

        iftWriteDataSet(originalDataset,"/home/peixinho/originalDataset.zip");
        iftWriteDataSet(datasetPreProcessing,"/home/peixinho/preprocessingDataset.zip");

        int count = iftCountSamples(artificialDataset, IFT_OUTLIER);

        qDebug() << "Count outliers: " << count;

        iftWriteOPFDataSet(artificialDataset, filename.toLatin1().data());
    }
}

void MainWindow::on_preProcessComboBox_currentIndexChanged(int index)
{
    applyLDA = applyPCA = false;

    switch (index) {
    case 1:
        applyPCA = true;
    case 2:
        applyLDA = true;
        break;
    default:
        break;
    }
}

void MainWindow::on_pushButton_clicked() {
    removeNodes();
}

void MainWindow::on_actionSave_dataset_triggered()
{
    saveDataset();
}
