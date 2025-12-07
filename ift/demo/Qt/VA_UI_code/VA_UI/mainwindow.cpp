#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsScene>

#include <QtWidgets>
#include "colorlisteditor.h"


void MainWindow::initComponents(){
    initMessageBox();
    initGraphicsViewProjectionArea();
    initListWidgetImagesThumb();
    initLabelImageCurrentSample();
    initTableWidgetSelectedSamples();
    initGraphWidgets();
    initTabWidget();
    initWidgetPlot();
    initComponentsConnections();
    initLocalVariables();

    //TODO
    drawingManager.drawOption = COLOR;
    drawingManager.objectOption = NONE;

    //drawingManager.nodeColorOption = COLOR_NONE;
    //drawingManager.nodeHiddenOption = HIDDEN_NONE;
}

void MainWindow::initMessageBox(){
    messageBox.setFixedSize(500,200);
}

//area(component) where the scene will be placed in
void MainWindow::initGraphicsViewProjectionArea(){
    ui->graphicsViewProjectionArea->setStyleSheet("border: 1px solid black");
    ui->graphicsViewProjectionArea->setDragMode(QGraphicsView::RubberBandDrag);
    ui->graphicsViewProjectionArea->setInteractive(true);
}
void MainWindow::initListWidgetImagesThumb(){
    ui->listWidgetImagesThumb->clear();
    ui->listWidgetImagesThumb->setStyleSheet("border: 1px solid black");
    ui->listWidgetImagesThumb->setViewMode(QListWidget::IconMode);
    ui->listWidgetImagesThumb->setIconSize(QSize(50,50));
    ui->listWidgetImagesThumb->setResizeMode(QListWidget::Adjust);
}

void MainWindow::initLabelImageCurrentSample(){
    ui->labelImageCurrentSample->clear();
    ui->labelImageCurrentSample->setStyleSheet("border: 1px solid black");
}

void MainWindow::initTableWidgetSelectedSamples(){
    ui->tableWidgetSelectedSamples->clear();
    ui->tableWidgetSelectedSamples->setStyleSheet("border: 1px solid black");
}

//scene
void MainWindow::initGraphWidgets(){
    graphWidget = new GraphWidget(ui->graphicsViewProjectionArea);
    graphWidget->setDragMode(QGraphicsView::RubberBandDrag);
}

void MainWindow::initTabWidget(){
    ui->tabWidget->setVisible(true);
}

void MainWindow::initWidgetPlot(){
    ui->widgetPlot->clearGraphs();
    createPlot();
    avaregeAccuracyOverLearningCycles.clear();
    standardDeviationOverLearningCycles.clear();
    avaregeAccuracyOverLearningCycles << 0;
    standardDeviationOverLearningCycles << 0;
}


void MainWindow::initLocalVariables(){
    colorForUnknowns = QColor(0,0,0);
    formDisplaySampleImage = NULL;
    dialogUpdateSamplesImageDirectory = NULL;
    directory = QString("/images");
    prefix = QString("sample");
    changes = false;

}

void MainWindow::initComponentsConnections(){
    connect(graphWidget->scene(),SIGNAL(selectionChanged())
            ,this,SLOT(updateSelectedSamples()));

    connect(graphWidget,SIGNAL(SelectedOptionSginal(DrawingManager)),
            this,SLOT(mouseCustomSceneMenu(DrawingManager)));

    connect(graphWidget,SIGNAL(mouseMoveCoordinates(QPointF* ,QPointF*)),
            this,SLOT(mouseMoveScene(QPointF*,QPointF* )));
}



MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    initComponents();
}

void MainWindow::setData(){



    QString fileName_dataset = QFileDialog::getOpenFileName(this, tr("Open File"), QString(),
                                                            tr("Dataset File (*.zip);;"));

    if(!fileName_dataset.isEmpty() && !fileName_dataset.isNull()){
        clearAllData();
        const char* path2Dataset  =  fileName_dataset.toLocal8Bit().constData();
        workDataset = loadDataSet(path2Dataset);
        resetDataSetLabels(workDataset);
        createDefaultClasses(workDataset);
        initProgressBar();
        progressDialog->setLabelText("Creating image Icons...");
        progressDialog->setValue(progressDialog->value()+1);
        creatIconsOnThumb();

        progressDialog->setLabelText(QString("Selector pre-prosseing"));
        progressDialog->setValue(progressDialog->value()+1);
        initActiveLearningManager(workDataset);
        iftCommonActiveLearningGoToNextState(activeLearning);//start
        iftCommonActiveLearningGoToNextState(activeLearning);//preprocessing
        createColorTableForClusters(workDataset);

        progressDialog->setLabelText(QString("Applying dimensionality reduction on dataset samples"));
        progressDialog->setValue(progressDialog->value()+1);
        if(tsne != NULL){
            iftDestroyTsne(&tsne);
        }
        tsne = iftCreateTsne(workDataset);
        iftComputeTsneProjection(tsne);
        datasetLowDimension = tsne->outputDataSet;
        iftMinMaxFeatureScale(datasetLowDimension);
        createGraphNodes(datasetLowDimension);

        drawNodesInSceneFirstTime(false);

        destroyProgressBar();
    }
}

iftDataSet* MainWindow::loadDataSet(const char *pathname){
    return iftReadOPFDataSet(pathname);
}

void MainWindow::resetDataSetLabels(iftDataSet* dataset){
    for (int sampleIndex = 0; sampleIndex < dataset->nsamples; ++sampleIndex) {
        dataset->sample[sampleIndex].label = 0;
    }
}

void MainWindow::initProgressBar(){
    progressDialog = new QProgressDialog(this);
    bar = new QProgressBar(progressDialog);
    progressDialog->setWindowModality(Qt::WindowModal);
    progressDialog->setCancelButton(NULL);
    bar->setRange(0, 5);
    bar->setValue(0);
    bar->setFormat("%v/%m");
    progressDialog->setBar(bar);
    progressDialog->setMinimumWidth(350);
    progressDialog->setMinimumDuration(200);
    progressDialog->setWindowModality(Qt::WindowModal);
    progressDialog->setValue(0);
}

void MainWindow::destroyProgressBar(){
    progressDialog->close();
    delete bar;
    delete progressDialog;

    bar = NULL;
    progressDialog = NULL;
}

void MainWindow::creatIconsOnThumb(iftDataSet* dataset){
    unsigned long long uniqueId;
    for (int i = 0; i < dataset->nsamples; ++i) {
        uniqueId = iftComputeUniqueIdForSample(dataset->sample[i]);
        hashsampleId2IconIndex.insert(uniqueId,i);
        //QString sampleName = mountSampleName(originalDataset->sample[i].id);
        QString sampleName = mountSampleName(&(workDataset->sample[i]));
        QListWidgetItem *itemWidget = new QListWidgetItem(QIcon("icons/imageMissing.png"),sampleName);
        itemWidget->setFlags(itemWidget->flags() & Qt::NoItemFlags);
        itemWidget->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEditable | Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
        ui->listWidgetImagesThumb->addItem(itemWidget);
//        hashsampleId2FileLocation.insert(uniqueId,"icons/imageMissing.png");
//        hashIconIndex2FileLocation.insert(i,"icons/imageMissing.png");
    }
}

void MainWindow::initActiveLearningManager(iftDataSet* dataset){
    if(activeLearning != NULL){
        iftDestroyCommonActiveLearning(&activeLearning);
        activeLearning = NULL;
    }
    activeLearning = iftCreateCommonActiveLearning();

    iftGenericClassifier *classifierOPFSUP = createGenericClassifier(CLASSIFIER_OPF_SUP);
    iftSelectorRDS* rds = createSelectorRDS(dataset);
    rds->classifier = classifierOPFSUP;
    iftGenericSelector* selectorRDS = createGenericSelector();
    selectorRDS->selector = (void*)rds;
    selectorRDS->freeFunctionSelector = iftDestroySelectorRDSVoidPointer;


    activeLearning->selector = selectorRDS;
    activeLearning->classifier = classifierOPFSUP;

    activeLearning->functionUsed2PreProcessing = iftPreprocessSelectorRDS;
    activeLearning->functionUsed2SelectSamples = iftSelecSamplesSelectorRDS;
}

void MainWindow::createColorTableForClusters(iftDataSet* dataset){
    int nclusters = dataset->nlabels;

    QColor color;
    double h = 0.0;
    double s = 1.0;
    double l = 0.7;
    double step = 1.0/nclusters;

    hashClusterId2ClusterColor.clear();
    hashClusterId2ClusterColor.insert(0,QColor(0,0,0));

    for (int i = 0; i < nclusters; ++i) {
        color.setHslF(h,s,l);
        h += step;
        hashClusterId2ClusterColor.insert(i+1,QColor(color));
    }
}

void MainWindow::drawNodesInSceneFirstTime(bool paintNodes){
    if(paintNodes == true){
        drawNodesTrueLabel();
    }else{
        drawNodesNone();
    }

}

void MainWindow::mouseMoveScene(QPointF* currentMousePositionInGraphicArea,
                                QPointF* currentMousePositionInScene){

    mouseCoordInGraphicArea.setX(currentMousePositionInGraphicArea->x());
    mouseCoordInGraphicArea.setY(currentMousePositionInGraphicArea->y());
    mouseCoordInScene.setX(currentMousePositionInScene->x());
    mouseCoordInScene.setY(currentMousePositionInScene->y());
    if(graphNodes.size() > 0){
        if(QGraphicsItem *item = graphWidget->itemAt(mouseCoordInGraphicArea.x(),mouseCoordInGraphicArea.y())){

            int nodeIndex = hashNodeItemPointer2GraphNodeIndex[item];
            QListWidgetItem* itemWidget = ui->listWidgetImagesThumb->item(nodeIndex);
            if(itemWidget == NULL){
                return;
            }
            QPixmap pic = itemWidget->icon().pixmap(ui->labelImageCurrentSample->width(),
                                                    ui->labelImageCurrentSample->height());
            pic = pic.scaled(ui->labelImageCurrentSample->width(),
                             ui->labelImageCurrentSample->height(),
                             Qt::KeepAspectRatio,
                             Qt::SmoothTransformation);
            ui->labelImageCurrentSample->setPixmap(pic);
        }
    }
}


MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::createPlot(){
    ui->widgetPlot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom );
    ui->widgetPlot->legend->setVisible(true);
    ui->widgetPlot->legend->setFont(QFont("Helvetica",8));
    ui->widgetPlot->legend->setMaximumSize(10,10);
    ui->widgetPlot->legend->setMinimumSize(80,40);
    //mean plot
    ui->widgetPlot->addGraph();
    ui->widgetPlot->graph(0)->setPen(QPen(Qt::blue));
    ui->widgetPlot->graph(0)->setLineStyle(QCPGraph::lsLine);
    ui->widgetPlot->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, 4));
    ui->widgetPlot->graph(0)->setName("Mean");
    ui->widgetPlot->axisRect()->setupFullAxesBox();
    ui->widgetPlot->replot();
    ui->widgetPlot->yAxis->setRange(0.0,1.0);
    ui->widgetPlot->xAxis->setRange(0.0,1.0);
    ui->widgetPlot->graph(0)->addData(0,0);

    //std plot
    ui->widgetPlot->addGraph();
    QPen pen;
    pen.setStyle(Qt::DotLine);
    pen.setWidth(1);
    pen.setColor(QColor(180,180,180));
    ui->widgetPlot->graph(1)->setName("standard deviation");
    ui->widgetPlot->graph(1)->setPen(pen);
    ui->widgetPlot->graph(1)->setBrush(QBrush(QColor(255,50,30,20)));
    ui->widgetPlot->addGraph();
    ui->widgetPlot->legend->removeItem(ui->widgetPlot->legend->itemCount()-1); // don't show two confidence band graphs in legend
    ui->widgetPlot->graph(2)->setPen(pen);
    ui->widgetPlot->graph(1)->setChannelFillGraph(ui->widgetPlot->graph(2));
    ui->widgetPlot->graph(1)->addData(0,0);
    ui->widgetPlot->graph(2)->addData(0,0);
}

void MainWindow::updatePlot(float mean, float std, int iteration){
    ui->widgetPlot->xAxis->setRange(0.0,avaregeAccuracyOverLearningCycles.size());
    ui->widgetPlot->graph(0)->addData(iteration,mean);
    ui->widgetPlot->graph(1)->addData(iteration,mean+std);
    ui->widgetPlot->graph(2)->addData(iteration,mean-std);
    ui->widgetPlot->replot();

}

void MainWindow::clearAllData(){
    clearAllHashTable();
    clearAllDatasets();
    clearAllGraphNodes();
    clearAllGraphEdges();
    clearPlotArea();
    clearThumbIcons();
}

void MainWindow::clearAllHashTable(){
    hashClusterId2ClusterColor.clear();
    hashIconIndex2FileLocation.clear();
    hashLabelId2LabelColor.clear();
    hashLabelId2LabelName.clear();
    hashNodeItemPointer2GraphNodeIndex.clear();
    hashsampleId2FileLocation.clear();
    hashsampleId2GraphNodeIndex.clear();
    hashsampleId2IconIndex.clear();
    hashsampleId2labelId.clear();
    hashsampleId2SupervisedSample.clear();
}

void MainWindow::clearAllDatasets(){
    if(workDataset != NULL){
        iftDestroyDataSet(&workDataset);
        workDataset = NULL;
    }
    if(datasetLowDimension != NULL){
        iftDestroyDataSet(&datasetLowDimension);
        datasetLowDimension = NULL;
    }
}

void MainWindow::clearAllGraphNodes(){
    graphNodes.clear();
}

void MainWindow::clearAllGraphEdges(){
    graphKnnMap.clear();
    graphPredMap.clear();
}

void MainWindow::clearPlotArea(){
    ui->widgetPlot->clearGraphs();
}

void MainWindow::clearThumbIcons(){
    ui->listWidgetImagesThumb->clear();
}






void MainWindow::createGraphNodes(iftDataSet *dataset2D){
    int x;
    int y;
    unsigned long long uniqueId;
    for(int i=0; i<dataset2D->nsamples; i++){
        Node *node = new Node(graphWidget);
        x = (dataset2D->sample[i].feat[0])*drawScaleWidthFactor + rootSampleDiameter;
        y = (dataset2D->sample[i].feat[1])*drawScaleHeightFactor + rootSampleDiameter;
        node->setPos(qreal(x),qreal(y));
        node->setFlag(QGraphicsItem::ItemIsSelectable);

        graphWidget->scene()->addItem(node);
        graphNodes.push_back(node);
        uniqueId = iftComputeUniqueIdForSample(dataset2D->sample[i]);
        hashsampleId2GraphNodeIndex.insert(uniqueId,i);
        hashNodeItemPointer2GraphNodeIndex.insert(node,i);
    }
}

void MainWindow::createGraphPredecessorEdgesPath(iftKnnGraph *opfClusteringGrapgh){
    for(int i=0; i<opfClusteringGrapgh->nnodes; i++){
        if(opfClusteringGrapgh->node[i].pred != IFT_NIL){
            int sourceNodeIndex = opfClusteringGrapgh->node[i].sample;
            int destinationNodeIndex = opfClusteringGrapgh->node[i].pred;
            Edge *edge = new Edge(graphNodes.at(sourceNodeIndex),
                                  graphNodes.at(destinationNodeIndex));
            edge->setEdgeType(predecessorEdge);
            graphWidget->scene()->addItem(edge);
            graphPredMap.push_back(edge);
            edge->setVisible(false);
        }
    }
}

void MainWindow::minMaxdataset(iftDataSet* dataset){
    float *minimus = (float*)iftAlloc(dataset->nfeats,sizeof(float));
    float *maximus = (float*)iftAlloc(dataset->nfeats,sizeof(float));

    for (int j = 0; j < dataset->nfeats; ++j) {
        minimus[j] = dataset->sample[0].feat[j];
        maximus[j] = dataset->sample[0].feat[j];
    }

    for (int j = 0; j < dataset->nfeats; ++j) {
        for (int i = 0; i < dataset->nsamples; ++i) {

            if(minimus[j] > dataset->sample[i].feat[j]){
                minimus[j] = dataset->sample[i].feat[j];
            }

            if(maximus[j] < dataset->sample[i].feat[j]){
                maximus[j] = dataset->sample[i].feat[j];
            }
        }
    }

    for (int j = 0; j < dataset->nfeats; ++j) {
        for (int i = 0; i < dataset->nsamples; ++i) {
            dataset->sample[i].feat[j] = (dataset->sample[i].feat[j] - minimus[j])/(maximus[j]-minimus[j]);
        }
    }
}


void MainWindow::resizeEvent(QResizeEvent* event)
{
    if(mainWindowInitHeight == 0 && mainWindowInitWidth == 0){
        mainWindowInitHeight = event->size().height();
        mainWindowInitWidth = event->size().width();
        mainWindowHeight = mainWindowInitHeight;
        mainWindowWidth = mainWindowInitWidth;
    }
    else{
        mainWindowHeight = event->size().height();
        mainWindowWidth = event->size().width();
        //int minDimension = iftMin(mainWindowHeight,mainWindowWidth);
        sceneProjectionAreaHeight = 0.75*mainWindowHeight;
        sceneProjectionAreaWidth = 0.75*mainWindowHeight;
        sceneProjectionAreaX = mainWindowWidth*0.025;
        sceneProjectionAreaY = 10;

        int selectedSamplesAreaX = sceneProjectionAreaX + sceneProjectionAreaWidth+3;
        int selectedSamplesAreaY = sceneProjectionAreaY;
        int selectedSamplesAreaWidth = sceneProjectionAreaWidth;
        int selectedSamplesAreaHeight = sceneProjectionAreaHeight+3;

        ui->graphicsViewProjectionArea->setGeometry(sceneProjectionAreaX,
                                                    sceneProjectionAreaY,
                                                    sceneProjectionAreaWidth+3,
                                                    sceneProjectionAreaHeight+3);

        //        ui->listWidgetSelectedSamples->setGeometry(sceneProjectionAreaX + sceneProjectionAreaWidth+3,
        //                                                   sceneProjectionAreaY,sceneProjectionAreaWidth,
        //                                                   sceneProjectionAreaHeight+3);
        //        ui->listWidgetSelectedSamples->setVisible(false);
        //        ui->listWidgetSelectedSamples->setEnabled(false);

        ui->tableWidgetSelectedSamples->setGeometry(0,
                                                    0,selectedSamplesAreaWidth,
                                                    selectedSamplesAreaHeight-ui->pushButton_2->height());



        ui->tabWidget->setGeometry(selectedSamplesAreaX,
                                   selectedSamplesAreaY,selectedSamplesAreaWidth,
                                   selectedSamplesAreaHeight-ui->pushButton_2->height());
        ui->tabWidget->setVisible(true);
        ui->widgetPlot->setGeometry(0,
                                    0,
                                    sceneProjectionAreaWidth,
                                    sceneProjectionAreaHeight-40);

        graphWidget->setSceneRect(0,
                                  0,
                                  sceneProjectionAreaWidth,
                                  sceneProjectionAreaHeight);
        graphWidget->setMinimumSize(sceneProjectionAreaWidth+2,
                                    sceneProjectionAreaHeight+2);


        ui->pushButton_2->setGeometry(selectedSamplesAreaX+selectedSamplesAreaWidth-ui->pushButton_2->width(),
                                      selectedSamplesAreaY+selectedSamplesAreaHeight-ui->pushButton_2->height(),
                                      ui->pushButton_2->width(), ui->pushButton_2->height());



        scrollAreaY = sceneProjectionAreaY+sceneProjectionAreaHeight+2;
        scrollAreaX = mainWindowWidth*0.025 + 100;

        scrollAreaHeight = mainWindowHeight-(sceneProjectionAreaHeight+sceneProjectionAreaY)-60;
        scrollAreaWidth = mainWindowWidth*0.95 - 100;

        drawScaleHeightFactor = sceneProjectionAreaHeight-(2*drawBorderDistance);
        drawScaleWidthFactor = sceneProjectionAreaWidth-(2*drawBorderDistance);

        drawBorderDistanceNormalized = drawBorderDistance/sceneProjectionAreaHeight;


        ui->listWidgetImagesThumb->setGeometry(scrollAreaX,scrollAreaY,scrollAreaWidth,scrollAreaHeight);
        ui->labelImageCurrentSample->setGeometry(scrollAreaX-100,scrollAreaY,
                                                 100,scrollAreaHeight);


    }
}

void MainWindow::on_listWidgetImagesThumb_doubleClicked(const QModelIndex &index)
{
    createWindowForIconInThumb(index);
}

void MainWindow::createWindowForIconInThumb(const QModelIndex &index){
    this->formDisplaySampleImage = new FormDisplaySampleImage(this,hashIconIndex2FileLocation[index.row()]);
    this->formDisplaySampleImage->setAttribute( Qt::WA_DeleteOnClose );
    this->formDisplaySampleImage->show();
}



void MainWindow::createWindowToSetDirectory(){

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

void MainWindow::updateSelectedSamples(){
    if(graphWidget->items().length()){

        for (int i = 0; i < graphPredMap.length(); ++i) {
            if(graphPredMap.at(i)->isVisible()){
                graphPredMap.at(i)->setVisible(false);
            }
        }
        for (int i = 0; i < graphKnnMap.length(); ++i) {
            if(graphKnnMap.at(i)->isVisible()){
                graphKnnMap.at(i)->setVisible(false);
            }
        }

        ui->listWidgetImagesThumb->setUpdatesEnabled(false);
        ui->listWidgetImagesThumb->blockSignals(true);
        for (int i = 0; i < graphNodes.length(); ++i) {
            if(graphNodes.at(i)->isSelected() && ui->listWidgetImagesThumb->item(i)->isSelected()){
                showPredecessorPath(graphNodes[i]);
                //showKnnNeighbours(graphNodes[i]);
            }else if(graphNodes.at(i)->isSelected() && !ui->listWidgetImagesThumb->item(i)->isSelected()){
                ui->listWidgetImagesThumb->item(i)->setSelected(true);
                ui->listWidgetImagesThumb->setCurrentRow(i);
                showPredecessorPath(graphNodes[i]);
                //showKnnNeighbours(graphNodes[i]);
            }else{
                ui->listWidgetImagesThumb->item(i)->setSelected(false);

            }
        }
        ui->listWidgetImagesThumb->setUpdatesEnabled(true);
        ui->listWidgetImagesThumb->blockSignals(false);
        ui->listWidgetImagesThumb->update();

    }
}

void MainWindow::showPredecessorPath(Node *node){
    for(int i=0;i<node->edges().length();i++){
        if(node->edges().at(i)->getEdgeType() == predecessorEdge){
            if(node != node->edges().at(i)->destNode()){
                if(!node->edges().at(i)->isVisible()){
                    node->edges()[i]->setVisible(true);
                    showPredecessorPath(node->edges()[i]->destNode());
                }
                return;
            }
        }
    }
}


void MainWindow::windowPrefixDirClosed(){
    delete this->dialogUpdateSamplesImageDirectory;
    dialogUpdateSamplesImageDirectory = NULL;
    if (changes) {
        if(workDataset != NULL){
            updateIconsOnThumb(originalDataset,true);
        }
    }
}

int* MainWindow::countNumberSamplesInClusters(iftKnnGraph *opfClusteringGrapgh,int nclusters){
    int* numberSamplesPerCluster = (int *)iftAlloc(nclusters+1,sizeof(int));
    int label;
    for(int i=0; i < opfClusteringGrapgh->Z->nsamples; i++){
        label = opfClusteringGrapgh->Z->sample[i].label;
        numberSamplesPerCluster[label]++;
    }
    return numberSamplesPerCluster;
}

void MainWindow::on_listWidgetImagesThumb_itemSelectionChanged()
{
    updateSceneFromThumb();

}

void MainWindow::updateSceneFromThumb(){
    graphWidget->scene()->blockSignals(true);
    int lastObjectSelectedIndex = -1;
    for (int i = 0; i < ui->listWidgetImagesThumb->count(); ++i) {
        if(ui->listWidgetImagesThumb->item(i)->isSelected() ){
            if((!graphNodes.at(i)->isSelected())){
                graphNodes.at(i)->setSelected(true);
                lastObjectSelectedIndex = i;

            }else{
                graphNodes.at(i)->setSelected(true);
            }
        }
        else{
            graphNodes.at(i)->setSelected(false);
        }
    }
    graphWidget->scene()->blockSignals(false);
    updateSelectedSamples();
    if(lastObjectSelectedIndex > -1){
        graphWidget->centerOn(graphNodes.at(lastObjectSelectedIndex));
    }
}


void MainWindow::on_pushButton_2_clicked()
{
    activeLearningFoward();
}

void MainWindow::activeLearningMainFlow(){
    if(workDataset == NULL){
        messageBox.information(0," ","Dataset is empty");
        return;
    }

    if(programStatus == LABELING_SAMPLES){
        int numSelected = (currentLearningCycle == 0 ? nClusters : 2 * nClusters);
        selectedSamples = iftPickFromUnlabeledSelector(activeSelector, numSelected, activeClassifier);
        highlightSelectedSamplesBySelector(selectedSamples);
        //prepareTable(selectedSamples);

        if(currentLearningCycle > 0){
            updateSampleColorForSuggestionLabel();
            ui->tabWidget->setCurrentIndex(0);
        }


    }else if(programStatus == TRAINING_CLASSIFIER){
        iftNode * listIter = selectedSamples->head;
        while (listIter != NULL) {
            int sample = listIter->elem;
            workDataset->sample[sample].status = IFT_TRAIN;
            hashsampleId2SupervisedSample[workDataset->sample[sample].id] = true;
            numberSupervisedSamples++;
            listIter = listIter->next;
        }
        unhighlightSelectedSamplesBySelector(selectedSamples);
        selectedSamplesHistory.append(selectedSamples);
        iftTrainActiveClassifier(activeClassifier);
        validate();
        ui->tabWidget->setCurrentIndex(1);

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

            updateSampleIcon(absolutePath,dataset->sample[i].id);
            percentage = ((float)i/list.size())*100;
            progressDialog.setValue(percentage);
        }
    }
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
    ui->listWidgetImagesThumb->item(iconIndex)->setIcon(QIcon(QPixmap::fromImage(imageResized)));
    hashsampleId2FileLocation[sampleId] = imageFileAbsolutePath;
    hashIconIndex2FileLocation[iconIndex] = imageFileAbsolutePath;
    listPaths2Images.append(imageFileAbsolutePath);
}


//QString MainWindow::mountSampleName(int sampleId){
//    QString sampleName = QString(prefix);
//    sampleName.append(QString::number(sampleId));
//    return sampleName;
//}

QString MainWindow::mountSampleName(iftSample* sample){
    char name[100];
    sprintf(name,"%06d_%08d.png",sample->truelabel,sample->id);
    QString sampleName = QString(name);
    return sampleName;
}


void MainWindow::findSampleFile(QFileInfoList directoryList, QString fileName, int sampleId){
    QStringList listString;

    for (int i = 0; i < directoryList.size(); ++i) {
        QFileInfo fileInfo = directoryList.at(i);
        listString = fileInfo.fileName().split('.');
        if(!listString.isEmpty()){

            if( !QString::compare(listString.at(0), fileName, Qt::CaseSensitive) ){
                QImage image(fileInfo.filePath());
                QImage imageResized = image.scaled(QSize(50,50));
                ui->listWidgetImagesThumb->item(sampleId)->setIcon(QIcon(QPixmap::fromImage(imageResized)));
                hashsampleId2FileLocation[sampleId] = fileInfo.absoluteFilePath();
                int iconIndex = hashsampleId2IconIndex[sampleId];
                hashIconIndex2FileLocation[iconIndex] = fileInfo.absoluteFilePath();
                break;
            }
        }
    }
}

void MainWindow::mouseCustomSceneMenu(DrawingManager selectedAction){

    drawingManager.drawOption = selectedAction.drawOption;
    drawingManager.objectOption = selectedAction.objectOption;

    if(drawingManager.drawOption == COLOR){
        drawNodes();
        currentDrawingNodeOption = drawingManager.objectOption;
    }else if(drawingManager.drawOption == HIDDEN){
        hiddenNodes();
    }else if(drawingManager.drawOption == LABEL_PROPAGATION){
        QComboBox *comboBox = new  QComboBox(ui->graphicsViewProjectionArea);
        int index = 0;
        for (int j = 0; j < hashLabelId2LabelName.size(); ++j) {
            comboBox->addItem(hashLabelId2LabelName[j], hashLabelId2LabelColor[j]);
            const QModelIndex idx = comboBox->model()->index(index++, 0);
            comboBox->model()->setData(idx, hashLabelId2LabelColor[j], Qt::DecorationRole);
        }
        comboBox->showPopup();
        connect(comboBox,SIGNAL(currentIndexChanged(int))
                ,this,SLOT(propagateLabel(int)));
        //  comboBox->deleteLater();
    }


}

void MainWindow::propagateLabel(int labelId){
    int R = hashLabelId2LabelColor[labelId].red();
    int G = hashLabelId2LabelColor[labelId].green();
    int B = hashLabelId2LabelColor[labelId].blue();
    for (int i = 0; i < graphNodes.size(); ++i) {
        if(graphNodes.at(i)->isSelected()){
            graphNodes.at(i)->setColor(R,G,B);
            graphNodes.at(i)->update();
            int sampleId = hashsampleId2GraphNodeIndex.key(i);
            //TODO: map sampleiIndex_inOiriginalDataser 2 sampleId
            workDataset->sample[sampleId].truelabel = labelId;
            workDataset->sample[sampleId].label = labelId;
            activeSelector->Z->sample[sampleId].truelabel = labelId;
            activeSelector->Z->sample[sampleId].label = labelId;
            hashsampleId2labelId[sampleId] = labelId;
        }
    }
}

void MainWindow::hiddenNodes(){
    if(workDataset == NULL){
        return;
    }

    if(drawingManager.objectOption == NONE){
        hiddenNoneNodes();
    }
    else if(drawingManager.objectOption == HIDDEN_UNSUPERVISED){
        hiddenUnsupervisedNodes();
    }


}

void MainWindow::hiddenNoneNodes(){
    for (int i = 0; i < workDataset->nsamples; ++i) {
        int sampleId = workDataset->sample[i].id;
        int nodeIndex = hashsampleId2GraphNodeIndex[sampleId];
        graphNodes.at(nodeIndex)->setVisible(true);
        graphNodes.at(nodeIndex)->update();
    }
}

void MainWindow::hiddenAllNodes(){
    for (int i = 0; i < workDataset->nsamples; ++i) {
        int sampleId = workDataset->sample[i].id;
        int nodeIndex = hashsampleId2GraphNodeIndex[sampleId];
        graphNodes.at(nodeIndex)->setVisible(false);
        graphNodes.at(nodeIndex)->update();
    }
}

void MainWindow::hiddenUnsupervisedNodes(){
    for (int i = 0; i < workDataset->nsamples; ++i) {
        int sampleId = workDataset->sample[i].id;
        int nodeIndex = hashsampleId2GraphNodeIndex[sampleId];
        if(!hashsampleId2SupervisedSample[nodeIndex] ){
            graphNodes.at(nodeIndex)->setVisible(false);
            graphNodes.at(nodeIndex)->update();
        }
    }
}

//void MainWindow::hiddenUnsupervisedNodes(){
//    for (int i = 0; i < originalDataset->nsamples; ++i) {
//        int sampleId = originalDataset->sample[sampleIndex].id;
//        int nodeIndex = hashsampleId2GraphNodeIndex[sampleId];
//        if(hashsampleId2SupervisedSample[sampleId]){

//        }
//        graphNodes.at(nodeIndex)->setVisible(false);
////        graphNodes.at(nodeIndex)->update();
//    }
//}


void MainWindow::highlightSelectedSamplesBySelector(iftList * selectedSamples){
    iftNode *node = selectedSamples->head;
    for (int i = 0; i < selectedSamples->n; ++i) {
        int sampleIndex = node->elem;

        if(currentDrawingNodeOption == COLOR_TRUELABELBASED){
            highlightNodeTrueLabel(sampleIndex);
        }
        else if(currentDrawingNodeOption == COLOR_OPFCLUSTERBASED){
            highlightNodeCluster(sampleIndex);
        }

        node = node->next;
    }
}

void MainWindow::unhighlightSelectedSamplesBySelector(iftList * selectedSamples){
    iftNode *node = selectedSamples->head;
    for (int i = 0; i < selectedSamples->n; ++i) {
        int sampleIndex = node->elem;

        if(currentDrawingNodeOption == COLOR_TRUELABELBASED){
            unhighlightNodeTrueLabel(sampleIndex);
        }
        if(currentDrawingNodeOption == COLOR_OPFCLUSTERBASED){
            unhighlightNodeCluster(sampleIndex);
        }

        node = node->next;
    }
}

void MainWindow::prepareTable(iftList * selectedSamples){
    iftNode *node = selectedSamples->head;
    int sampleid;

    ui->tableWidgetSelectedSamples->clear();
    ui->tableWidgetSelectedSamples->setRowCount(selectedSamples->n);
    ui->tableWidgetSelectedSamples->setColumnCount(2);
    //    ui->tableWidgetSelectedSamples->setHorizontalHeaderLabels(QStringList() << tr("Selected sample")
    //                                                              << tr("suggested label") << tr("label"));
    ui->tableWidgetSelectedSamples->setHorizontalHeaderLabels(QStringList() << tr("Selected sample")
                                                              << tr("Suggeted Label"));

    ui->tableWidgetSelectedSamples->verticalHeader()->setVisible(false);
    comboBoxes.clear();
    QSignalMapper* signalMapper = new QSignalMapper(this);
    for (int i = 0; i < selectedSamples->n; ++i) {
        QComboBox *cb = new QComboBox(ui->tableWidgetSelectedSamples);
        int index = 0;
        for (int j = 0; j < hashLabelId2LabelName.size(); ++j) {
            cb->addItem(hashLabelId2LabelName[j], hashLabelId2LabelColor[j]);
            const QModelIndex idx = cb->model()->index(index++, 0);
            cb->model()->setData(idx, hashLabelId2LabelColor[j], Qt::DecorationRole);
        }
        comboBoxes << cb;

        connect(cb,SIGNAL(currentIndexChanged(int))
                ,signalMapper,SLOT(map()));
        signalMapper->setMapping(cb, QString("%1").arg(i));


        sampleid = workDataset->sample[node->elem].id;
        int suggestedLabelId = activeSelector->Z->sample[node->elem].label;
        workDataset->sample[node->elem].label = suggestedLabelId;
        hashsampleId2labelId[sampleid] = suggestedLabelId;

        cb->setCurrentIndex(suggestedLabelId);


        QIcon icon(QPixmap(hashsampleId2FileLocation[sampleid]));
        QTableWidgetItem *iconItem = new QTableWidgetItem();
        QImage *image = new QImage;
        image->load(hashsampleId2FileLocation[sampleid]);
        iconItem->setData(Qt::DecorationRole,QPixmap::fromImage(*image));

        hashsampleId2SupervisedSample[sampleid] = true;
        int labelId = hashsampleId2labelId[sampleid];
        QColor color= hashLabelId2LabelColor[labelId];
        QTableWidgetItem *colorItem = new QTableWidgetItem();
        colorItem->setData(Qt::DisplayRole,color);

        QPixmap pic ((hashsampleId2FileLocation[sampleid]));
        QPixmap pic2 = pic.scaled(50,50,Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
        QLabel *labelIcon = new QLabel;
        labelIcon->setPixmap(pic2);
        labelIcon->setAlignment(Qt::AlignCenter);

        ui->tableWidgetSelectedSamples->setCellWidget(i,0,labelIcon);
        //ui->tableWidgetSelectedSamples->setCellWidget(i,1,suggestedLabelName);
        ui->tableWidgetSelectedSamples->setCellWidget(i,1,cb);

        node = node->next;
    }
    connect(signalMapper, SIGNAL(mapped(const QString &)),
            this, SLOT(updateColorOfCurrentSelectedSamples(const QString &)));



    ui->tableWidgetSelectedSamples->horizontalHeader()->setDefaultSectionSize(50);
    ui->tableWidgetSelectedSamples->verticalHeader()->setDefaultSectionSize(40);
    ui->tableWidgetSelectedSamples->resizeColumnToContents(1);
    ui->tableWidgetSelectedSamples->horizontalHeader()->setStretchLastSection(true);

}



void MainWindow::loadCategories(){
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open Categories"), QString(),
                                                    tr("All files (*)"));
    bool insertUnknownCategorie = true;
    if (!fileName.isEmpty()) {
        if(updateHashLabelName(fileName,insertUnknownCategorie)){
            updateHashLabelColor();

            hashLabelId2LabelColor[0] = colorForUnknowns;
        };
        messageBox.information(0,"Load Labels","Labels were loaded");
    }
}

bool orderPairByAlphabeticalOrder(const QPair<int,QString> &v1, const QPair<int,QString> &v2)
{
    return v1.second < v2.second;
}

bool orderPairByTrueLabelOrder(const QPair<int,QString> &v1, const QPair<int,QString> &v2)
{
    return v1.first < v2.first;
}



bool MainWindow::updateHashLabelName(QString fileName_categories, bool insertUnknownCategorie){
    QFile inputFile(fileName_categories);
    bool sucess = false;

    if (inputFile.open(QIODevice::ReadOnly))
    {
        QTextStream in(&inputFile);
        int i=0;
        hashLabelId2LabelName.clear();
        if(insertUnknownCategorie){
            hashLabelId2LabelName.insert(i,"unknown");
            i++;
        }
        while (!in.atEnd())
        {
            QStringList stringList= in.readLine().split(" ");
            QPair< int, QString > pair;
            QString numberString = stringList.at(0);
            pair.first = numberString.toInt();
            pair.second = stringList.at(1);

            listPairTrueLabelIdAndTrueLabelName.append(pair);
            i++;
        }

        //alphabetical order
        qSort( listPairTrueLabelIdAndTrueLabelName.begin(),
               listPairTrueLabelIdAndTrueLabelName.end(), orderPairByAlphabeticalOrder);
        for (int var = 0; var <  listPairTrueLabelIdAndTrueLabelName.size(); ++var) {
            hashLabelId2LabelName.insert(listPairTrueLabelIdAndTrueLabelName.at(var).first,
                                         listPairTrueLabelIdAndTrueLabelName.at(var).second);
        }
        inputFile.close();
        sucess = true;
    }
    else{
        sprintf(erroMessagesBuffer,"[updateHashLabelName] Could not open file %s", fileName_categories.toLatin1().data());
        messageBox.critical(0,"Error",erroMessagesBuffer);
        sucess = false;
    }
    return sucess;
}




void MainWindow::updateHashLabelColor(){
    /*This function assumes hashLabelId2LabelName is updated*/
    hashLabelId2LabelColor.clear();
    double h = 0.0;
    double s = 1.0;
    double v = 1.0;
    double step = 1.0/hashLabelId2LabelName.size();

    QColor color;

    hashLabelId2LabelName.insert(0,"unknown");
    hashLabelId2LabelColor.insert(0,QColor(0,0,0));

    for (int i = 0; i < hashLabelId2LabelName.size(); ++i) {
        color.setHsvF(h,s,v);
        h += step;
        hashLabelId2LabelColor.insert(i+1,QColor(color));
    }
}



void MainWindow::updateColorOfCurrentSelectedSamples(const QString &comboBoxRow){
    int comboBoxRow_int = comboBoxRow.toInt();
    QComboBox* combo = (QComboBox*)ui->tableWidgetSelectedSamples->
            cellWidget(comboBoxRow_int,1);
    int labelIndex = combo->currentIndex();
    int sampleId = -1;
    int nodeIndex = -1;
    iftNode *node;
    node = selectedSamples->head;
    for (int i = 0; i < selectedSamples->n; ++i) {
        if(i != comboBoxRow_int){
            node = node->next;
        }else{
            sampleId = workDataset->sample[node->elem].id;
            nodeIndex = hashsampleId2GraphNodeIndex[sampleId];
            hashsampleId2labelId[sampleId] = labelIndex;
            break;
        }
    }

    int R = hashLabelId2LabelColor[labelIndex].red();
    int G = hashLabelId2LabelColor[labelIndex].green();
    int B = hashLabelId2LabelColor[labelIndex].blue();
    graphNodes.at(nodeIndex)->setColor(R,G,B);
    graphNodes.at(nodeIndex)->update();
    workDataset->sample[sampleId].truelabel = labelIndex;
    workDataset->sample[sampleId].label = labelIndex;

    QLabel *labelIcon = (QLabel*)ui->tableWidgetSelectedSamples->cellWidget(comboBoxRow_int,0);
    QString style = QString("border: 2px solid rgb(%1,%2,%3)").arg(R).arg(G).arg(B);
    labelIcon->setStyleSheet(style);
}

void MainWindow::updateSampleColorForSuggestionLabel(){
    iftNode *node;
    node = selectedSamples->head;
    for (int i = 0; i < selectedSamples->n; ++i) {
        int label = activeSelector->Z->sample[node->elem].label;
        int sampleId = workDataset->sample[node->elem].id;
        int nodeIndex = hashsampleId2GraphNodeIndex[sampleId];
        int R = hashLabelId2LabelColor[label].red();
        int G = hashLabelId2LabelColor[label].green();
        int B = hashLabelId2LabelColor[label].blue();
        graphNodes.at(nodeIndex)->setColor(R,G,B);
        graphNodes.at(nodeIndex)->update();
        workDataset->sample[node->elem].truelabel = label;
        workDataset->sample[node->elem].label = label;

        QLabel *labelIcon = (QLabel*)ui->tableWidgetSelectedSamples->cellWidget(i,0);
        QString style = QString("border: 2px solid rgb(%1,%2,%3)").arg(R).arg(G).arg(B);
        labelIcon->setStyleSheet(style);
        node = node->next;
    }
}


void MainWindow::prepareDatasetTest(){
    for (int i = 0; i < datasetTest->nsamples; ++i) {
        bool supervised = hashsampleId2SupervisedSample[datasetTest->sample[i].id];
        if(supervised){
            datasetTest->sample[i].status =  IFT_TEST;
            datasetTest->sample[i].label =  0;
        }else{
            datasetTest->sample[i].status =  IFT_UNKNOWN;
        }
    }
}

iftDataSet* MainWindow::datasetDimensionalityReduction(iftDataSet *dataset){
    iftTsne* tsne = iftCreateTsne(dataset);
    tsne->max_iter = 500;
    tsne->stop_lying_iter = 50;
    tsne->mom_switch_iter = 125;
    tsne->theta = 0.5;
    tsne->learningRate = 200;
    iftComputeTsneProjection(tsne);


    return tsne->outputDataSet;
}


void MainWindow::drawNodesInScene(){
    //drawNodesNone();
    drawNodesTrueLabel();
}



void MainWindow::drawNodes(){
    if(workDataset == NULL){
        return;
    }

    else if(drawingManager.objectOption == COLOR_TRUELABELBASED){
        drawNodesTrueLabel();
    }else if(drawingManager.objectOption == COLOR_LABELBASED){
        drawNodesLabel();
    }
    else if(drawingManager.objectOption == COLOR_OPFCLUSTERBASED){
        drawNodesCluster();
    }
}

void MainWindow::drawNodesNone(){
    unsigned long long uniqueId;
    for (int i = 0; i < workDataset->nsamples; ++i) {
        uniqueId = iftComputeUniqueIdForSample(workDataset->sample[i]);
        int nodeIndex = hashsampleId2GraphNodeIndex[uniqueId];
        graphNodes.at(nodeIndex)->setColor(0,0,0);
        graphNodes.at(nodeIndex)->update();
    }
}

void MainWindow::drawNodesLabel(){
    if(!alreadyComputedLabelsForAllSamples){
        for (int i = 0; i < workDataset->nsamples; ++i) {
            iftActiveClassifierClassifySample(activeClassifier, workDataset, i);
            int sampleId = workDataset->sample[i].id;
            int nodeIndex = hashsampleId2GraphNodeIndex[sampleId];
            int labelId = workDataset->sample[i].label;
            QColor color = hashLabelId2LabelColor[labelId];
            graphNodes.at(nodeIndex)->setColor(color.red(),color.green(),color.blue());
            graphNodes.at(nodeIndex)->update();
        }

        alreadyComputedLabelsForAllSamples = true;
    }else{
        for (int i = 0; i < workDataset->nsamples; ++i) {
            int sampleId = workDataset->sample[i].id;
            int nodeIndex = hashsampleId2GraphNodeIndex[sampleId];
            int labelId = workDataset->sample[i].label;
            QColor color = hashLabelId2LabelColor[labelId];
            graphNodes.at(nodeIndex)->setColor(color.red(),color.green(),color.blue());
            graphNodes.at(nodeIndex)->update();
        }
    }
}

void MainWindow::drawNodesTrueLabel(){
    unsigned long long uniqueId;
    for (int i = 0; i < workDataset->nsamples; ++i) {
        uniqueId = iftComputeUniqueIdForSample(workDataset->sample[i]);
        int nodeIndex = hashsampleId2GraphNodeIndex[uniqueId];
        int labelId = workDataset->sample[i].truelabel;

        QColor color = hashLabelId2LabelColor[labelId];
        qDebug() << uniqueId << nodeIndex << labelId << color.red() << color.green() << color.blue();
        graphNodes.at(nodeIndex)->setColor(color.red(),color.green(),color.blue());
        graphNodes.at(nodeIndex)->update();
    }
}

void MainWindow::drawNodesCluster(){
    for (int i = 0; i < workDataset->nsamples; ++i) {
        int sampleId = workDataset->sample[i].id;
        int nodeIndex = hashsampleId2GraphNodeIndex[sampleId];
        int clusterId =  workDataset->sample[i].cluster_id;
        QColor color = hashClusterId2ClusterColor[clusterId];
        graphNodes.at(nodeIndex)->setColor(color.red(),color.green(),color.blue());
        graphNodes.at(nodeIndex)->update();
    }
}


void MainWindow::highlightNodeTrueLabel(int sampleIndex){
    int sampleId = workDataset->sample[sampleIndex].id;
    int sampleLabel = workDataset->sample[sampleIndex].label;
    int nodeIndex = hashsampleId2GraphNodeIndex[sampleId];
    QColor color = hashLabelId2LabelColor[sampleLabel];
    graphNodes.at(nodeIndex)->setVisible(true);
    graphNodes.at(nodeIndex)->setColor(color.red(),color.green(),color.blue());
    graphNodes.at(nodeIndex)->setColorBorder(R_borderSelected,G_borderSelected,B_borderSelected);
    graphNodes.at(nodeIndex)->selectedByClassifier = true;
    graphNodes.at(nodeIndex)->setZValue(1);
    graphNodes.at(nodeIndex)->setRadius(rootSampleDiameter/2);
    graphNodes.at(nodeIndex)->update();
}

void MainWindow::unhighlightNodeTrueLabel(int sampleIndex){
    int sampleId = workDataset->sample[sampleIndex].id;
    int nodeIndex = hashsampleId2GraphNodeIndex[sampleId];
    graphNodes.at(nodeIndex)->setColorBorder(0,0,0);
    graphNodes.at(nodeIndex)->setZValue(0);
    graphNodes.at(nodeIndex)->setRadius(commonSampleDiameter/2);
    graphNodes.at(nodeIndex)->update();
}

void MainWindow::highlightNodeCluster(int sampleIndex){
    int sampleId = workDataset->sample[sampleIndex].id;
    int clusterId = workDataset->sample[sampleIndex].cluster_id;
    int nodeIndex = hashsampleId2GraphNodeIndex[sampleId];
    QColor color = hashClusterId2ClusterColor[clusterId];
    graphNodes.at(nodeIndex)->setVisible(true);
    graphNodes.at(nodeIndex)->setColor(color.red(),color.green(),color.blue());
    graphNodes.at(nodeIndex)->setColorBorder(R_borderSelected,G_borderSelected,B_borderSelected);
    graphNodes.at(nodeIndex)->selectedByClassifier = true;
    graphNodes.at(nodeIndex)->setZValue(1);
    graphNodes.at(nodeIndex)->setRadius(rootSampleDiameter/2);
    graphNodes.at(nodeIndex)->update();
}

void MainWindow::unhighlightNodeCluster(int sampleIndex){
    int sampleId = workDataset->sample[sampleIndex].id;
    int nodeIndex = hashsampleId2GraphNodeIndex[sampleId];
    graphNodes.at(nodeIndex)->setColorBorder(0,0,0);
    graphNodes.at(nodeIndex)->setZValue(-1);
    graphNodes.at(nodeIndex)->setRadius(commonSampleDiameter/2);
    graphNodes.at(nodeIndex)->update();
}

void MainWindow::validate(){
    iftDataSet *testDataSet = iftCreateDataSet(numberSupervisedSamples,workDataset->nfeats);
    testDataSet->nclasses = hashLabelId2LabelName.size()-1;
    int k = 0;
    for (int i = 0; i < workDataset->nsamples; ++i) {
        int sampleId = workDataset->sample[i].id;
        if(hashsampleId2SupervisedSample[sampleId]){
            iftCopySample(&(workDataset->sample[i]),
                          &(testDataSet->sample[k]),
                          workDataset->nfeats,
                          true);
            k++;
        }
    }
    iftSampler* kfold = iftNKFold(numberSupervisedSamples, 5,2);
    float *acc  = iftAllocFloatArray(kfold->niters);
    iftCplGraph  *graph=NULL;

    for (int i=0; i < kfold->niters; i++) {
        iftSampleDataSet(testDataSet, kfold, i);

        graph = iftSupLearn(testDataSet);// Execute the supervised learning
        iftClassify(graph,testDataSet);                // Classify test samples in Z
        acc[i] = iftNormAccuracy(testDataSet);     // Compute accuracy on test set
        iftDestroyCplGraph(&graph);

    }

    float mean = iftMean(acc,kfold->niters);
    float stdev = iftStddevFloatArray(acc,kfold->niters);
    avaregeAccuracyOverLearningCycles << mean;
    standardDeviationOverLearningCycles << stdev;

    if(ui->widgetPlot->plottableCount() == 0){
        createPlot();
    }
    updatePlot(mean, stdev, currentLearningCycle);
    iftDestroySampler(&kfold);
    iftDestroyDataSet(&testDataSet);
    free(acc);
}





void MainWindow::on_actionDataSet_triggered()
{
    resetTrueLabels = true;
    setData();
}

void MainWindow::on_actionCategories_2_triggered()
{
    loadCategories();
}

void MainWindow::on_actionDataset_Images_Directory_2_triggered()
{
    createWindowToSetDirectory();
}

void MainWindow::on_actionSave_dataset_triggered()
{
    saveCurrentDataSetLabels();
}

void MainWindow::saveCurrentDataSetLabels(){
    if(workDataset == NULL){
        sprintf(erroMessagesBuffer,"[saveCurrentDataSetLabels] Dataset is empty");
        messageBox.warning(0,"Warning", erroMessagesBuffer);
        return;
    }
    QString filename = QFileDialog::getSaveFileName(this,"Save file","",".zip");
    filename.append(".zip");
    if(filename.isEmpty()){
        return;
    }
    if(!alreadyComputedLabelsForAllSamples){
        for (int i = 0; i < workDataset->nsamples; ++i) {
            iftActiveClassifierClassifySample(activeClassifier, workDataset, i);
        }
        alreadyComputedLabelsForAllSamples = true;
    }
    iftWriteOPFDataSet(workDataset,filename.toLatin1().data());
}


void MainWindow::createDefaultClasses(iftDataSet* dataset){

    if(dataset->nclasses == 0){
        iftCountNumberOfClassesDataSet(dataset);
    }
    int nclasses = dataset->nclasses;

    QColor color;
    double h = 0.0;
    double s = 1.0;
    double v = 1.0;
    double step = 1.0/nclasses;

    hashLabelId2LabelColor.clear();
    hashLabelId2LabelName.clear();
    hashLabelId2LabelName.insert(0,"unknown");
    hashLabelId2LabelColor.insert(0,QColor(0,0,0));

    for (int i = 0; i < nclasses; ++i) {
        color.setHsvF(h,s,v);
        h += step;
        hashLabelId2LabelColor.insert(i+1,QColor(color));
        QString trueLabelName = "class" + QString::number(i+1);
        hashLabelId2LabelName.insert(i+1,trueLabelName);
    }



}

iftDataSet* MainWindow::createDatasetFromList(QList<iftSample*> list,int nfeats){
    iftDataSet* dataset = iftCreateDataSet(list.size(),nfeats);
    for (int var = 0; var < list.size();   var++) {
        iftCopySample(list.at(var),&(dataset->sample[var]),nfeats,true);
    }
    return dataset;
}

void MainWindow::activeLearningFoward(){

    switch (programStatus) {
    case BEGIN:
        programStatus = LABELING_SAMPLES;
        break;
    case LABELING_SAMPLES:
        if(currentLearningCycle == 0){
            programStatus = LABELING_SAMPLES;
        }else{
            programStatus = TRAINING_CLASSIFIER;
        }
        currentLearningCycle++;
        break;
    case TRAINING_CLASSIFIER:
        programStatus = LABELING_SAMPLES;
        break;
    default:
        break;
    }
    activeLearningMainFlow();
}

