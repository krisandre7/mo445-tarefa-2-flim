#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "mywidgetitem.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QCloseEvent>
#include <QMouseEvent>

//#include <QDir>



MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    treeWidgetSamples = ui->treeWidget;
    treeWidgetSamples->setSelectionMode(QAbstractItemView::ExtendedSelection);
    treeWidgetSamples->header()->setStretchLastSection(false);
    myGraphicWidgetProjection = ui->graphicsView;
    myGraphicWidgetProjection->setCacheMode(QGraphicsView::CacheBackground);
    myGraphicWidgetProjection->setViewportUpdateMode(QGraphicsView::SmartViewportUpdate);
    selectedSamplesIndicesVec.clear();


    initScene();
    initGraphicsView();
    initColorNodeOptionsList();
    initInfoArea();
    initPlot();
    initiWindowSceneOptions();

    //connect(windowSceneOptions,SIGNAL(sceneAutoProjectionChanged(bool)),ui->graphicsView,SLOT(updateSceneAutoReprojection(bool)));
    ui->graphicsView->autoReprojection = windowSceneOptions->sceneAutomaticReprojection;

}

void MainWindow::initiWindowSceneOptions(){
    windowSceneOptions = new DialogSceneOptions(this,developingMode);
    windowSceneOptions->setHidden(true);


    windowSceneOptions->objectSize = pointDefaultRx;
    windowSceneOptions->objectSize_spinBox->setValue(pointDefaultRx);

    connect(windowSceneOptions,SIGNAL(windowClosed()),this,SLOT(windowSceneOptionsClosed()));
    connect(windowSceneOptions,SIGNAL(nodeColorPaintChanged()),this,SLOT(windowNodeColorPaintChanged()));
    connect(windowSceneOptions,SIGNAL(edgeOptionChanged()),this,SLOT(windowEdgeOptionChanged()));
    connect(windowSceneOptions,SIGNAL(labelPropagationChanged(int)),this,SLOT(windowLabelPropagationChanged(int)));
    connect(windowSceneOptions,SIGNAL(filterClusterChanged(int)),this,SLOT(windowFilterClusterChanged(int)));
    connect(windowSceneOptions,SIGNAL(filterClassChanged(int)),this,SLOT(windowFilterClassChanged(int)));
    connect(windowSceneOptions,SIGNAL(filterClassChanged(int)),this,SLOT(windowFilterClassChanged(int)));
    connect(windowSceneOptions,SIGNAL(objectSizeEditFinished(double)),this,SLOT(windowObjectSizeChanged(double)));
    connect(windowSceneOptions,SIGNAL(constrastValueChanged(int)),this,SLOT(windowConstrastChanged(int)));
    connect(windowSceneOptions,SIGNAL(brightValueChanged(int)),this,SLOT(windowBrightChanged(int)));
    connect(windowSceneOptions,SIGNAL(binaryOptionChanged()),this,SLOT(windowBinaryOptionChanged()));
    connect(windowSceneOptions,SIGNAL(shapeOptionChanged(int)),this,SLOT(windowShapeOptionChanged(int)));
    connect(windowSceneOptions,SIGNAL(sceneManualReproject()),this,SLOT(windowManualReprojectChanged()));

    currentNodeColorOption = windowSceneOptions->currentPaintNodeOption;
}

void MainWindow::initVectors(){
    int maximumNumberOfCycles = 150;
    accTrueLabelVec.resize(maximumNumberOfCycles);
    accSmallDataset.resize(maximumNumberOfCycles);
    accStdSmallDataset.resize(maximumNumberOfCycles);
    accNormTrueLabelVec.resize(maximumNumberOfCycles);
    accNormSmallDataset.resize(maximumNumberOfCycles);
    accNormStdSmallDataset.resize(maximumNumberOfCycles);
    kappaTrueLabelVec.resize(maximumNumberOfCycles);
    kappaSmallDataset.resize(maximumNumberOfCycles);
    kappaStdSmallDataset.resize(maximumNumberOfCycles);

    selectedSamplesIndicesVecHistory.resize(maximumNumberOfCycles);
    numberOfSupervisedSamplesHistory.resize(maximumNumberOfCycles);
    numberOfLabelPropagatedSamplesHistory.resize(maximumNumberOfCycles);
    accuracy.resize(maximumNumberOfCycles);
    accuracyNorm.resize(maximumNumberOfCycles);
    cohenKappa.resize(maximumNumberOfCycles);
    learnCycleValues.resize(maximumNumberOfCycles);
    operationNumber.resize(maximumNumberOfCycles);
}

void MainWindow::initGraphicsView(){
    ui->graphicsView->setDragMode(QGraphicsView::RubberBandDrag);
    ui->graphicsView->setInteractive(true);

    //MyGraphicWidget* aux= dynamic_cast<MyGraphicWidget*>(ui->graphicsView);
    connect(ui->graphicsView, SIGNAL(myMouseContextMenuSignal(QContextMenuEvent *)), this, SLOT(createMouseMenuForPropagation(QContextMenuEvent *)) );
    connect(ui->graphicsView, SIGNAL(sceneAreaSizeChanged()), this, SLOT(setNodesPosition()) );
}

void MainWindow::createMouseMenuForPropagation(QContextMenuEvent *event){
    Q_UNUSED(event);
    QComboBox *comboBox = new  QComboBox(ui->graphicsView);
    int index = 0;
    for (int j = 0; j < hashLabelId2LabelName.size(); ++j) {
        comboBox->addItem(hashLabelId2LabelName[j], hashLabelId2LabelColor[j]);
        const QModelIndex idx = comboBox->model()->index(index++, 0);
        comboBox->model()->setData(idx, hashLabelId2LabelColor[j], Qt::DecorationRole);
    }
    comboBox->setGeometry(event->x(),event->y(),400,400);
    comboBox->showPopup();
    connect(comboBox,SIGNAL(currentIndexChanged(int))
            ,this,SLOT(windowLabelPropagationChanged(int)));
}

void MainWindow::setNodesPosition(){
    if(workDataset == NULL){
        return;
    }
    if(graphNodes.size() <= 0){
        return;
    }
    if(scene == NULL){
        return;
    }
    qreal x,y;
    qreal rx = pointDefaultRx;
    qreal ry = pointDefaultRy;
    qreal scalinFactor_width = (myGraphicWidgetProjection->size().width()-(rx*2));
    qreal scalinFactor_height = (myGraphicWidgetProjection->size().height()-(ry*2));
    qreal factor;
    if(scalinFactor_height < scalinFactor_width){
        factor = scalinFactor_height;
    }else{
        factor = scalinFactor_width;
    }

    for (int i = 0; i < graphNodes.size(); ++i) {
        x = (workDataset->projection->val[i*2])*factor  + rx;
        y = (workDataset->projection->val[i*2 + 1])*factor + ry;
        MyNode* node = graphNodes.at(i);
        node->setPos(x,y);
        node->update();
    }

}

void MainWindow::initPlot(){
    if(plot != NULL){
        plot->clearGraphs();
        delete plot;
    }
    plot = new QCustomPlot();
    plot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom );
    plot->legend->setVisible(true);
    plot->legend->setFont(QFont("Helvetica",8));
    //plot->legend->setMaximumSize(10,10);
    plot->legend->setMinimumSize(80,25);
    //mean plot
    plot->addGraph();
    plot->graph(0)->setPen(QPen(Qt::blue));
    plot->graph(0)->setLineStyle(QCPGraph::lsLine);
    plot->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, 4));
    plot->graph(0)->setName("Acc");
    plot->axisRect()->setupFullAxesBox();
    plot->replot();
    plot->yAxis->setRange(0.0,1.0);
    plot->xAxis->setRange(0.0,1.0);
    plot->resize(420, 340);
}

void MainWindow::initScene(){
    if(scene != NULL){
        scene->clear();
        delete scene;
    }
    scene = new QGraphicsScene(this);
    ui->graphicsView->setScene(scene);
}

void MainWindow::initColorNodeOptionsList(){

    if(developingMode == true){
        nodeColorOptions << "Class";
        nodeColorOptions << "Cluster";
        nodeColorOptions << "Sup + Class";
        nodeColorOptions << "Predicted Class";
        nodeColorOptions << "Cluster RDS";
    }
    nodeColorOptions << "text cluster sup";
    nodeColorOptions << "classifier prediction";
    nodeColorOptions << "text cluster sup+propagated";
    nodeColorOptions << "heatmap_distance";
    nodeColorOptions << "heatmap_geodesic_distance";
    nodeColorOptions << "heatmap_geodesic_distance_normalized";
    nodeColorOptions << "heatmap_in_frontier";

    binaryOptions << "none";
    binaryOptions << "cluster_frontier";
    binaryOptions << "root_mismatch";

}

void  MainWindow::initInfoArea(){
    selectedSamplesIndicesVec.clear();
    numberOfSupervisedSamplesHistory.clear();
    accuracy.clear();
    learnCycleValues.clear();
    selectedSamplesIndicesVecHistory.clear();
    //    selectedSamplesIndicesVec << 0;
    //    numberOfSupervisedSamplesHistory << 0;
    //    accuracy << 0;
    //    learnCycleValues << 0;
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::clearAllData(){
    ui->treeWidget->clear();
    scene->clear();
    scene->update();
    graphNodes.clear();
    graphEdges.clear();

    clearAllHashTable();
    clearVectors();
    clearLists();
    clearAllDatasets();
    clearActiveLearning();
    //    clearAllGraphNodes();
    //    clearAllGraphEdges();
    //    clearPlotArea();
    //    clearThumbIcons();
}

void MainWindow::clearAllHashTable(){
    hashClusterId2ClusterColor.clear();
    hashClusterId2ClusterName.clear();
    hashLabelId2LabelColor.clear();
    hashLabelId2LabelName.clear();
    hashGraphNodeRef2SampleId.clear();
    hashPath2SampleId.clear();
    hashSampleId2CurrentItemInList.clear();
    hashSampleId2CurrentSelected.clear();
    hashSampleId2DatasetRow.clear();
    hashSampleId2GraphNodeRef.clear();
}

void MainWindow::clearVectors(){
    labelsColor.clear();
    labelsNames.clear();
}

void MainWindow::clearLists(){
    ui->listWidget->clear();
    ui->listWidget_2->clear();
    ui->listWidget_2->allItemsHaveIcon = false;
    ui->listWidget_2->itemRefs.clear();
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
    if(testDataset != NULL){
        iftDestroyDataSet(&testDataset);
        testDataset = NULL;
    }
}

void MainWindow::clearActiveLearning(){
    if(activeLearning != NULL){
        iftDestroyCommonActiveLearning(&activeLearning);
        activeLearning = NULL;
    }
}

void MainWindow::on_actionOpen_Dataset_triggered()
{
    openDataSetAndUpdateData();
}

void MainWindow::resetDataSetLabels(iftDataSet* dataset){
    for (int sampleIndex = 0; sampleIndex < dataset->nsamples; ++sampleIndex) {
        dataset->sample[sampleIndex].label = 0;
    }
}

void MainWindow::createDefaultClasses(iftDataSet* dataset){
    if(dataset == NULL){
        return;
    }
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
    hashLabelId2LabelColor.insert(0,QColor(0,0,0,200));
    for (int i = 0; i < nclasses; ++i) {
        color.setHsvF(h,s,v);
        h += step;
        color.setAlpha(200);
        hashLabelId2LabelColor.insert(i+1,QColor(color));
        QString trueLabelName = "class" + QString::number(i+1);
        hashLabelId2LabelName.insert(i+1,trueLabelName);
    }
}

void MainWindow::openDataSetAndUpdateData()
{
    QString fileName_dataset = QFileDialog::getOpenFileName(this, tr("Open Dataset"), "/home/deangeli/base_parasites/Base_+4",
                                                            tr("Dataset File (*.zip);;"));
    if(!fileName_dataset.isEmpty() && !fileName_dataset.isNull()){
        clearAllData();
        const char* path2Dataset  =  fileName_dataset.toLocal8Bit().constData();
        workDataset = iftReadDataSet(path2Dataset);
        bool alreadySupervisd = false;
        for (int i = 0; i < workDataset->nsamples; ++i) {
            if(workDataset->sample[i].isSupervised == true || (workDataset->sample[i].isLabelPropagated & 0b10000000) > 0){
                alreadySupervisd = true;
                break;
            }
        }


        initVectors();
        treeWidgetSamples->clear();
        if(workDataset->ref_data_type == IFT_REF_DATA_FILESET){
            fileSet = (iftFileSet*)workDataset->ref_data;
        }
        printf("\n**Dataset info**\n");
        iftPrintDataSetInfo(workDataset,false);
        fflush(stdout);
        initSelectors(workDataset);

        createMapBetweenRowsAndIds(workDataset);
        createDefaultClasses(workDataset);
        createColorTableForClusters(workDataset);
        createNamesForClusters(workDataset);



        MyThread* threadProjection = new MyThread(this,THREAD_INIT_GRAPH);
        threadProjection->start(QThread::HighestPriority);
        connect(threadProjection,SIGNAL(myFinished(int)),this,SLOT(threadFinished(int)));
        connect(threadProjection,SIGNAL(finished()),threadProjection,SLOT(deleteLater()));

        MyThread* threadTreeCreator = new MyThread(this,THREAD_INIT_TREEVIEW);
        qRegisterMetaType<QVector<int> >("QVector<int>");
        threadTreeCreator->start(QThread::HighPriority);
        connect(threadTreeCreator,SIGNAL(myFinished(int)),this,SLOT(threadFinished(int)));
        connect(threadTreeCreator,SIGNAL(finished()),threadTreeCreator,SLOT(deleteLater()));

        createVectorWithLabelsName2FillComboBox();
        createVectorWithLabelsColor2FillComboBox();
        createRepresentativeSamplesElementsInList(workDataset);
        updateInfoArea();


        mountConfusionMatrixWidget();
        confusionMatrixSmallDatasetMean = iftCreateMatrix(workDataset->nclasses,workDataset->nclasses);
        confusionMatrixSmallDatasetStandardDeviation = iftCreateMatrix(workDataset->nclasses,workDataset->nclasses);
        windowSceneOptions->fillPropagationComboBox(hashLabelId2LabelName,hashLabelId2LabelColor);
        windowSceneOptions->fillClassFilterComboBox(hashLabelId2LabelName,hashLabelId2LabelColor);
        windowSceneOptions->fillClusterFilterComboBox(hashClusterId2ClusterName,hashClusterId2ClusterColor);

        if(alreadySupervisd == false){
            resetDataSetLabels(workDataset);
        }else{
            for (int i = 0; i < workDataset->nsamples; ++i) {
                workDataset->sample[i].status = IFT_TEST;
                if(workDataset->sample[i].isSupervised == true){
                    workDataset->sample[i].status |= IFT_TRAIN;
                }
            }
            if(workDataset == NULL){
                QMessageBox::critical(this,"Error","Dataset is empty");
                return;
            }
            if(activeLearning == NULL){
                QMessageBox::critical(this,"Error","Active learner is empty");
                return;
            }

            int nt = iftCountSamples(workDataset,IFT_TRAIN);
            int nt2 = iftCountSamples(workDataset,IFT_TEST);
            if(nt == 0 || nt2 == 0){
                QMessageBox::critical(this,"Error","There are not train/test samples");
                return;
            }
            iftTrainGenericClassifier(activeLearning->classifier,workDataset);
            setTrainAndTestSamples();
            iftPredictGenericClassifier(activeLearning->classifier,workDataset);
            computeCrossValidationOnSupervisedSamples();
            if(developingMode == true){
                computeTrueLabelInfo();
            }
            updateTableWidget();
            numberOfSupervisedSamplesHistory[activeLearning->currentLearningCycle] = nt;
            learnCycleValues[activeLearning->currentLearningCycle] = activeLearning->currentLearningCycle;
            updateInfoArea();



        }
    }

}

void MainWindow::updateTestDataset(){
    for (int i = 0; i < testDataset->nsamples; ++i) {
        testDataset->sample[i].status = IFT_TEST;
    }
    for (int i = 0; i < workDataset->nsamples; ++i) {
        int id = workDataset->sample[i].id;
        testDataset->sample[id].status = IFT_UNKNOWN;
    }
    countTest = iftCountSamples(testDataset,IFT_TEST);
}

void MainWindow::initSelectors(iftDataSet* dataset){
    classifier = iftCreateGenericClassifier(CLASSIFIER_OPF_SUP);
    selectorRDS = iftCreateSelectorRDS(dataset);
    selectorRDS->classifier = classifier;
    iftPreProcessSelectorRDS(selectorRDS);

    printf("\nSelector Info\n");
    printf("**selector info**\n");
    iftPrintSelectorInfoRDS(selectorRDS,true);
    fflush(stdout);
    selector = iftCreateGenericSelector();
    selector->selecSamplesSelectorFunction = iftSelecSamplesByRDSVoidPointer;
    selector->selector =(void*)selectorRDS;
    activeLearning = iftCreateCommonActiveLearning();
    activeLearning->classifier = classifier;
    activeLearning->selector = selector;
}

void MainWindow::createMapBetweenRowsAndIds(iftDataSet* dataset){
    for (int i = 0; i < dataset->nsamples; ++i) {
        hashSampleId2DatasetRow.insert(dataset->sample[i].id,i);
    }
}

void MainWindow::updateInfoArea(){
    QString value;
    value = QString::number(learnCycleValues[activeLearning->currentLearningCycle]);
    ui->labelCurrentCycleValue->setText(value);

    if(seeTrueLabelInfo == true){
        value = QString::number(accTrueLabelVec[activeLearning->currentLearningCycle],10,4);
    }else{
        value = QString::number(accSmallDataset[activeLearning->currentLearningCycle],10,4);
    }
    ui->labelLastAccuracyValue->setText(value);

    if(seeTrueLabelInfo == true){
        value = QString::number(0,10,4);
    }else{
        value = QString::number(accStdSmallDataset[activeLearning->currentLearningCycle],10,4);
    }
    ui->labelLastAccuracyValueStd->setText(value);

    if(seeTrueLabelInfo == true){
        value = QString::number(accNormTrueLabelVec[activeLearning->currentLearningCycle],10,4);
    }else{
        value = QString::number(accNormSmallDataset[activeLearning->currentLearningCycle],10,4);
    }
    ui->labelLastAccuracyNormValue->setText(value);

    if(seeTrueLabelInfo == true){
        value = QString::number(0,10,4);
    }else{
        value = QString::number(accNormStdSmallDataset[activeLearning->currentLearningCycle],10,4);
    }
    ui->labelLastAccuracyNormValueStd->setText(value);

    if(seeTrueLabelInfo == true){
        value = QString::number(kappaTrueLabelVec[activeLearning->currentLearningCycle],10,4);
    }else{
        value = QString::number(kappaSmallDataset[activeLearning->currentLearningCycle],10,4);
    }
    ui->labelcohenKappa->setText(value);

    if(seeTrueLabelInfo == true){
        value = QString::number(0,10,4);
    }else{
        value = QString::number(kappaStdSmallDataset[activeLearning->currentLearningCycle],10,4);
    }
    ui->labelcohenKappaStd->setText(value);

    value = QString::number(selectedSamplesIndicesVec.size());
    ui->labelNumberOfSelectedSamples->setText(value);

    value = QString::number(numberOfSupervisedSamplesHistory[activeLearning->currentLearningCycle]);
    ui->labelNumberOfSupervisedSamplesValue->setText(value);

    value = QString::number(numberOfLabelPropagatedSamplesHistory[activeLearning->currentLearningCycle]);
    ui->labelNumberOfLabelPropagationValue->setText(value);

    //    if(numberOfSupervisedSamplesHistory.size() >= 0){
    //        value = QString::number(numberOfSupervisedSamplesHistory[activeLearning->currentLearningCycle]);
    //    }else{
    //        value = "0";
    //    }
    //    ui->labelNumberOfSupervisedSamplesValue->setText(value);

}

QString MainWindow::mountSampleName(iftSample* sample){
    char name[100];
    sprintf(name,"%06d_%08d.png",sample->truelabel,sample->id);
    QString sampleName = QString(name);
    return sampleName;
}

void MainWindow::createVectorWithLabelsName2FillComboBox(){
    if(hashLabelId2LabelName.size() <= 0){
        return;
    }

    for (int i = 0; i < hashLabelId2LabelName.size(); ++i) {
        labelsNames << hashLabelId2LabelName[i];
    }
}

void MainWindow::createVectorWithLabelsColor2FillComboBox(){
    if(hashLabelId2LabelColor.size() <= 0){
        return;
    }
    for (int i = 0; i < hashLabelId2LabelColor.size(); ++i) {
        labelsColor << hashLabelId2LabelColor[i];
    }
}

void MainWindow::createRepresentativeSamplesElementsInList(iftDataSet* dataset){
    if(dataset == NULL){
        return;
    }
    if(dataset->nclasses == 0){
        iftCountNumberOfClassesDataSet(dataset);
    }
    size_t nclasses = dataset->nclasses;
    for (size_t i = 0; i < nclasses; ++i) {
        MyWidgetItem* listItem = new MyWidgetItem(":/resource_images/Images/imageMissing.png",labelsNames,labelsColor,ui->listWidget_2);
        listItem->myComboBox->setCurrentIndex(i+1);
        ui->listWidget_2->addMyItem(listItem);
    }
    ui->listWidget_2->setEnableComboBoxes(false);
    ui->listWidget_2->backGroundSelected = QColor(255,255,255,0);
    ui->listWidget_2->listType = ReceiverType;
    ui->listWidget_2->buddyList = ui->listWidget;
    ui->listWidget->buddyList = ui->listWidget_2;
    connect(ui->listWidget_2,SIGNAL(propagateLabel(int)), ui->listWidget, SLOT(labelPropagated(int) ) );
    connect(ui->listWidget,SIGNAL(removedIndices(QVector<int>)), this, SLOT(updateDisplayOfSelectedSamples_removed(QVector<int>))  );

    if(marginOscilator_forBugCorrection == 1){
        marginOscilator_forBugCorrection = -1;
    }else{
        marginOscilator_forBugCorrection = 1;
    }
    ui->listWidget_2->setGeometry(ui->listWidget_2->geometry().x(),ui->listWidget_2->geometry().y(),
                                  ui->listWidget_2->geometry().width(),ui->listWidget_2->geometry().height()+marginOscilator_forBugCorrection);
}

void MainWindow::createTreeViewForOrganization(iftDataSet*  dataset,
                                               iftGenericVector* clusterOrganization,
                                               QTreeWidget* treeWidget){
    if(clusterOrganization != NULL){
        int sampleIndex;
        char name[20];
        int id;
        for (size_t clusterIndex = 0; clusterIndex < clusterOrganization->size; ++clusterIndex) {
            QTreeWidgetItem *treeItem = new QTreeWidgetItem(treeWidget);
            treeItem->setText(0,tr("clusterId"));
            treeItem->setText(1,tr("name"));
            iftGenericVector* cluster = iftVectorAt(iftGenericVector*,
                                                    clusterOrganization,
                                                    clusterIndex);

            for (size_t i = 0; i < cluster->size; ++i) {
                sampleIndex = iftVectorAt(int,cluster,i);
                QTreeWidgetItem *treeItemChild = new QTreeWidgetItem(treeItem);
                treeItemChild->setText(0,QString::number(clusterIndex));
                sprintf(name,"%06d_%08d",dataset->sample[sampleIndex].truelabel,dataset->sample[sampleIndex].id);
                treeItemChild->setText(1,QString(name));
                id = dataset->sample[sampleIndex].id;
                hashSampleId2TreeItem.insert(id,treeItemChild);
                hashTreeItem2SampleId.insert(treeItemChild,id);
            }
        }
    }
}

void MainWindow::updateSelectedItems_Tree2Graph(QTreeWidgetItem *item, int column){
    Q_UNUSED(item);
    Q_UNUSED(column);
    int id;
    if(hashTreeItem2SampleId.size() <= 0){
        return;
    }
    if(hashSampleId2GraphNodeRef.size() <= 0){
        return;
    }
    treeWidgetSamples->blockSignals(true);
    scene->blockSignals(true);
    QList<QGraphicsItem *> selecteditems = scene->selectedItems();
    for (int i = 0; i < selecteditems.size(); ++i) {
        selecteditems.at(i)->setSelected(false);
    }
    QList<QTreeWidgetItem *>treeItems = treeWidgetSamples->selectedItems();
    QTreeWidgetItem * treeItem;
    for (int i = 0; i < treeItems.size(); ++i) {
        treeItem = treeItems.at(i);
        id = hashTreeItem2SampleId[treeItem];
        MyNode* node = hashSampleId2GraphNodeRef[id];
        if(node == NULL){
            continue;
        }
        node->setSelected(true);
    }
    scene->blockSignals(false);
    treeWidgetSamples->blockSignals(false);
}

void MainWindow::createColorTableForClusters(iftDataSet* dataset){
    int nclusters = dataset->ngroups;
    QColor color;
    double h = 0.0;
    double s = 1.0;
    double l = 0.7;
    double step = 1.0/nclusters;
    hashClusterId2ClusterColor.clear();
    hashClusterId2ClusterName.clear();
    hashClusterId2ClusterColor.insert(0,QColor(0,0,0));
    for (int i = 0; i < nclusters; ++i) {
        color.setHslF(h,s,l);
        h += step;
        hashClusterId2ClusterColor.insert(i+1,QColor(color));
    }
}

void MainWindow::createNamesForClusters(iftDataSet* dataset){
    int nclusters = dataset->ngroups;
    hashClusterId2ClusterName.insert(0,"unknown");
    for (int i = 0; i < nclusters; ++i) {
        QString clusterName = "cluster" + QString::number(i+1);
        hashClusterId2ClusterName.insert(i+1,clusterName);
    }
}

void MainWindow::updateSelectedItems_Graph2Tree(){
    if(hashSampleId2TreeItem.size() <= 0 ){
        return;
    }
    QList<QTreeWidgetItem *> treeItems = treeWidgetSamples->selectedItems();
    for (int i = 0; i < treeItems.size(); ++i) {
        QTreeWidgetItem *treeItem = treeItems.at(i);
        treeItem->setSelected(false);
    }
    int id;
    for (int i = 0; i < scene->selectedItems().size(); ++i) {
        MyNode* item = dynamic_cast<MyNode*>(scene->selectedItems().at(i));
        id = hashGraphNodeRef2SampleId[item];
        if(id == 0){
            continue;
        }
        QTreeWidgetItem *treeItemChild = hashSampleId2TreeItem[id];
        treeItemChild->setSelected(true);
    }

}

void MainWindow::updateImages(iftDataSet* dataset){
    if(dataset == NULL){
        return;
    }
    if(hashSampleId2GraphNodeRef.size() <= 0){
        return;
    }

    QString absolutePath;
    int id;
    int trueLabel;
    if(fileSet == NULL){
        QMessageBox::warning(this,"warning","fileSet is not defined");
        for (int i = 0; i < dataset->nsamples; ++i) {
            id = dataset->sample[i].id;
            trueLabel = dataset->sample[i].truelabel;
            absolutePath = QString::number(trueLabel) + "|" + QString::number(id);
            MyNode* node = hashSampleId2GraphNodeRef[id];
            //sprintf(node->text,"%03d",trueLabel);
            node->sampleImagePath = "";
            node->sampleName = absolutePath;
            node->mountToopTip();
        }
    }else{
        for (int i = 0; i < dataset->nsamples; ++i) {
            id = dataset->sample[i].id;
            absolutePath = fileSet->files[id]->path;
            MyNode* node = hashSampleId2GraphNodeRef[id];
            QImage image(absolutePath);
            if(image.isNull()){
                node->sampleImagePath = ":/resource_images/Images/imageMissing.png";
                node->sampleName = absolutePath;
            }else{
                node->sampleImagePath = absolutePath;
                node->sampleName = absolutePath;
            }
            node->mountToopTip();
        }
    }
}

void MainWindow::threadFinished(int taskId){
    if(THREAD_INIT_GRAPH == taskId){
        for (int i = 0; i < graphNodes.size(); ++i) {
            scene->addItem(graphNodes.at(i));
        }
        connect(scene,SIGNAL( selectionChanged() ),this,  SLOT(updateSelectedItems_Graph2Tree()) );
        scene->setSceneRect(0,0,sceneScalingFactor,sceneScalingFactor);
        scene->update();
        currentNodeColorOption = windowSceneOptions->currentPaintNodeOption;
        paintNode(currentNodeColorOption);
    }else if(THREAD_INIT_TREEVIEW == taskId){
        connect(treeWidgetSamples,SIGNAL( itemClicked(QTreeWidgetItem *,int) ),this,SLOT(updateSelectedItems_Tree2Graph(QTreeWidgetItem *, int) ));
        updateTreeItemsToolTips(workDataset);
    }
}

void MainWindow::updateTreeItemsToolTips(iftDataSet* dataset){
    QString text;
    QString sampleImagePath;
    QString sampleName;
    iftFileSet* fileSet = (iftFileSet*)dataset->ref_data;
    int id;
    for (int i = 0; i < dataset->nsamples; ++i) {
        QTreeWidgetItem* item = hashSampleId2TreeItem[dataset->sample[i].id];
        if(item == NULL){
            continue;
        }
        if(dataset->ref_data != NULL && dataset->ref_data_type == IFT_REF_DATA_FILESET){
            id = dataset->sample[i].id;
            sampleImagePath = fileSet->files[id]->path;
            sampleName =  QString::number(id);
            text = "<html><img height=\"60\" src="+sampleImagePath+" /><br/>" +sampleName + "</html>";
        }else{
            id = dataset->sample[i].id;
            sampleName =  QString::number(id);
            text = sampleName;
        }
        for (int j = 0; j < item->columnCount(); ++j) {
            item->setToolTip(j,text);
        }
    }
}


void MainWindow::windowNodeColorPaintChanged(){
    if(workDataset == NULL || graphNodes.size() <= 0){
        return;
    }
    currentNodeColorOption = windowSceneOptions->currentPaintNodeOption;
    paintNode(currentNodeColorOption);
}

void MainWindow::windowEdgeOptionChanged(){
    if(workDataset == NULL || graphNodes.size() <= 0){
        return;
    }
    currentEdgeOption = windowSceneOptions->currentEdgeOption;
    setDrawEdges(currentEdgeOption);
}

void MainWindow::windowBinaryOptionChanged(){
    if(workDataset == NULL || graphNodes.size() <= 0){
        return;
    }
    currentBinaryOption = windowSceneOptions->currentBinaryOption;
    setDrawNodeBinary(currentBinaryOption);
}

void MainWindow::windowShapeOptionChanged(int shapeOption){
    if(workDataset == NULL || graphNodes.size() <= 0){
        return;
    }
    DrawingOption currentDrawOption;
    if(shapeOption == 0){
        currentDrawOption = optionDrawing;
    }else if(shapeOption == 1){
        currentDrawOption = optionText;
    }else if(shapeOption == 2){
        currentDrawOption = optionImage;
    }else{
        currentDrawOption = optionText;
    }

    for (int i = 0; i < graphNodes.size(); ++i) {
        MyNode* node = graphNodes.at(i);
        node->drawOption = currentDrawOption;
    }
    paintNode(currentNodeColorOption);
}

void MainWindow::windowManualReprojectChanged(){
    reprojectVisualization();
}

void MainWindow::windowLabelPropagationChanged(int labelIdPropagated){
    currentLabel = labelIdPropagated;
    if(workDataset == NULL){
        return;
    }
    if(hashSampleId2GraphNodeRef.isEmpty()){
        return;
    }

    for (int i = 0; i < workDataset->nsamples; ++i) {
        MyNode* node = hashSampleId2GraphNodeRef[workDataset->sample[i].id];
        if(node == NULL){
            continue;
        }
        if(node->isSelected() && workDataset->sample[i].isSupervised == false){
            workDataset->sample[i].truelabel = currentLabel;
            workDataset->sample[i].label = currentLabel;
            if(currentLabel == 0){
                //workDataset->sample[i].isSupervised = false;
                workDataset->sample[i].isLabelPropagated = false;
            }else{
                //workDataset->sample[i].isSupervised = true;
                workDataset->sample[i].isLabelPropagated = true;
            }
            node->fillColor = hashLabelId2LabelColor[currentLabel];
            node->update();
        }
    }
    trainClassifier();
    if(currentNodeColorOption == "classifier prediction"){
        paintNode(currentNodeColorOption);
    }
}

void MainWindow::windowFilterClusterChanged(int clusterIndexFiltred){
    filterCluster = clusterIndexFiltred;
    MyThread* threadProjection = new MyThread(this,THREAD_SEE_ONLY);
    threadProjection->start(QThread::HighPriority);
    connect(threadProjection,SIGNAL(myFinished(int)),this,SLOT(threadFinished(int)));
    connect(threadProjection,SIGNAL(finished()),threadProjection,SLOT(deleteLater()));
    paintNode(currentNodeColorOption);
}

void MainWindow::windowFilterClassChanged(int classIndexFiltred){
    filterClass = classIndexFiltred;
    MyThread* threadProjection = new MyThread(this,THREAD_SEE_ONLY);
    threadProjection->start(QThread::HighPriority);
    connect(threadProjection,SIGNAL(myFinished(int)),this,SLOT(threadFinished(int)));
    connect(threadProjection,SIGNAL(finished()),threadProjection,SLOT(deleteLater()));
    paintNode(currentNodeColorOption);
}

void MainWindow::windowObjectSizeChanged(double objectSize){

    if(graphNodes.size() <= 0 || workDataset == NULL){
        return;
    }

    for (int i = 0; i < graphNodes.size(); ++i) {
        MyNode* node = graphNodes.at(i);
        node->radiusX = objectSize;
        node->radiusY = objectSize;
    }
    paintNode(currentNodeColorOption);
}

void MainWindow::windowConstrastChanged(int contrast){
    qreal saturation= contrast;
    saturation = saturation/100;
    for (int i = 1; i < hashLabelId2LabelColor.size(); ++i) {
        hashLabelId2LabelColor[i].setHsvF(
                    hashLabelId2LabelColor[i].hueF(),
                    saturation,
                    hashLabelId2LabelColor[i].valueF());
    }
    paintNode(currentNodeColorOption);
}

void MainWindow::windowBrightChanged(int bright){
    qreal _bright = bright;
    _bright = _bright/100;
    for (int i = 1; i < hashLabelId2LabelColor.size(); ++i) {
        hashLabelId2LabelColor[i].setHsvF(
                    hashLabelId2LabelColor[i].hueF(),
                    hashLabelId2LabelColor[i].saturationF(),
                    _bright);
    }
    paintNode(currentNodeColorOption);
}

//void MainWindow::updateComboBoxPropgation(){
//    ui->comboBox_3->clear();
//    ui->comboBox_3->setCurrentIndex(0);
//    QComboBox* comboBox = ui->comboBox_3;
//    int index = 0;
//    for (int i = 0; i < hashLabelId2LabelName.size(); ++i) {
//        comboBox->addItem(hashLabelId2LabelName[i],hashLabelId2LabelColor[i]);
//        const QModelIndex idx = comboBox->model()->index(index++, 0);
//        comboBox->model()->setData(idx,hashLabelId2LabelColor[i], Qt::DecorationRole);
//    }
//}


void MainWindow::paintNode(QString comboBoxOption){
    if(comboBoxOption == "Class Color Supervised"){
        paintNodeSupervisionedClass(workDataset);
    }else if(comboBoxOption == "Class Color"){
        paintNodeClass(workDataset);
    }else if(comboBoxOption == "Predicted Class"){
        paintNodePredictedClass(workDataset);
    }else if(comboBoxOption == "Cluster"){
        paintNodeCluster(workDataset);
    }else if(comboBoxOption == "Text cluster sup"){
        paintNodeTextClusterSup(workDataset);
    }else if(comboBoxOption == "classifier prediction"){
        paintNodeTextClusterAll(workDataset);
    }else if(comboBoxOption == "Text cluster sup+propagated"){
        paintNodeTextClusterSupPropagated(workDataset);
    }else if(comboBoxOption == "Heatmap_distance"){
        iftSelectorRDS* rds = (iftSelectorRDS*)activeLearning->selector->selector;
        paintNodeTextClusterHeatMap_distance(workDataset,rds->clusterOfSamplesIndex,
                                             rds->samplesDistance2Root);
    }else if(comboBoxOption == "Samples_frontier_cluster"){
        iftSelectorRDS* rds = (iftSelectorRDS*)activeLearning->selector->selector;
        paintNodeTextClusterFrontier(rds->opfClusteringGrapgh);
    }else if(comboBoxOption == "Heatmap_geodesic_distance"){
        iftSelectorRDS* rds = (iftSelectorRDS*)activeLearning->selector->selector;
        paintNodeTextClusterHeatMap_geodesicDistance(rds->opfClusteringGrapgh);
    }else if(comboBoxOption == "Heatmap_geodesic_distance_normalized"){
        iftSelectorRDS* rds = (iftSelectorRDS*)activeLearning->selector->selector;
        paintNodeTextClusterHeatMap_geodesicDistanceNormalized(rds->opfClusteringGrapgh);
    }else if(comboBoxOption == "Heatmap_in_frontier"){
        iftSelectorRDS* rds = (iftSelectorRDS*)activeLearning->selector->selector;
        paintNodeTextClusterHeatMap_inFrontier(rds->opfClusteringGrapgh);
    }
}


void MainWindow::propagateLabels(int label){
    if(workDataset == NULL){
        return;
    }
    if(currentNodeColorOption == nodeColorOptions.at(0)){
        for (int i = 0; i < workDataset->nsamples; ++i) {
            MyNode* node = hashSampleId2GraphNodeRef[workDataset->sample[i].id];
            if(node == NULL){
                continue;
            }
            if(node->isSelected()){
                workDataset->sample[i].truelabel = label;
                workDataset->sample[i].label = label;
                //workDataset->sample[i].isSupervised = true;
            }
        }
        paintNodeSupervisionedClass(workDataset);
    }else if(currentNodeColorOption == nodeColorOptions.at(1)){
        for (int i = 0; i < workDataset->nsamples; ++i) {
            MyNode* node = hashSampleId2GraphNodeRef[workDataset->sample[i].id];
            if(node == NULL){
                continue;
            }
            if(node->isSelected()){
                workDataset->sample[i].truelabel = label;
                workDataset->sample[i].label = label;
            }
        }
        paintNodeClass(workDataset);
    }else if(currentNodeColorOption == nodeColorOptions.at(2)){
        for (int i = 0; i < workDataset->nsamples; ++i) {
            MyNode* node = hashSampleId2GraphNodeRef[workDataset->sample[i].id];
            if(node == NULL){
                continue;
            }
            if(node->isSelected()){
                workDataset->sample[i].label = label;
                workDataset->sample[i].label = label;
            }
        }
        paintNodePredictedClass(workDataset);
    }else if(currentNodeColorOption == nodeColorOptions.at(3)){
        for (int i = 0; i < workDataset->nsamples; ++i) {
            MyNode* node = hashSampleId2GraphNodeRef[workDataset->sample[i].id];
            if(node == NULL){
                continue;
            }
            if(node->isSelected()){
                workDataset->sample[i].group = label;
                workDataset->sample[i].label = label;
            }
        }
        paintNodeCluster(workDataset);
    }
}

void MainWindow::paintNodeSupervisionedClass(iftDataSet* dataset){
    if(hashSampleId2GraphNodeRef.size() <= 0 ){
        return;
    }
    int id;
    for (int sampleIndex = 0; sampleIndex < dataset->nsamples; ++sampleIndex) {
        id = dataset->sample[sampleIndex].id;
        MyNode* node = hashSampleId2GraphNodeRef[id];
        if(node == NULL){
            continue;
        }
        if(dataset->sample[sampleIndex].isSupervised){
            QColor color = hashLabelId2LabelColor[dataset->sample[sampleIndex].truelabel];
            node->fillColor = color;
            node->text = QString("%1").arg(dataset->sample[sampleIndex].truelabel);
        }else{
            QColor color(0,0,0,255);
            node->fillColor = color;
            node->text = "";
            node->isMarked = false;
        }
    }
    scene->update();
}

void MainWindow::paintNodeClass(iftDataSet* dataset){
    if(hashSampleId2GraphNodeRef.size() <= 0){
        return;
    }
    int id;
    for (int sampleIndex = 0; sampleIndex < dataset->nsamples; ++sampleIndex) {
        id = dataset->sample[sampleIndex].id;
        MyNode* node = hashSampleId2GraphNodeRef[id];
        if(node == NULL){
            continue;
        }
        QColor color = hashLabelId2LabelColor[dataset->sample[sampleIndex].truelabel];
        node->fillColor = color;
        node->text = QString("%1").arg(dataset->sample[sampleIndex].truelabel);
        node->isMarked = false;
    }
    scene->update();
}

void MainWindow::paintNodePredictedClass(iftDataSet* dataset){
    if(hashSampleId2GraphNodeRef.size() <= 0){
        return;
    }
    int id;
    for (int sampleIndex = 0; sampleIndex < dataset->nsamples; ++sampleIndex) {
        id = dataset->sample[sampleIndex].id;
        MyNode* node = hashSampleId2GraphNodeRef[id];
        if(node == NULL){
            continue;
        }
        QColor color = hashLabelId2LabelColor[dataset->sample[sampleIndex].label];
        node->fillColor = color;
        node->text = QString("%1").arg(dataset->sample[sampleIndex].label);
        node->isMarked = false;
    }
    scene->update();
}

void MainWindow::paintNodeCluster(iftDataSet* dataset){
    if(hashSampleId2GraphNodeRef.size() <= 0){
        return;
    }
    int id;
    for (int sampleIndex = 0; sampleIndex < dataset->nsamples; ++sampleIndex) {
        id = dataset->sample[sampleIndex].id;
        MyNode* node = hashSampleId2GraphNodeRef[id];
        if(node == NULL){
            continue;
        }
        QColor color = hashClusterId2ClusterColor[dataset->sample[sampleIndex].group];
        node->fillColor = color;
        node->text = QString("%1").arg(dataset->sample[sampleIndex].group);
        node->isMarked = false;
    }
    scene->update();
}

void MainWindow::paintNodeTextClusterHeatMap_distance(iftDataSet* dataset,iftGenericVector* clusterOfSamplesIndex,float** clusterDistance){
    Q_UNUSED(dataset);
    float maxDist = -1;
    for (size_t clusterIndex = 0; clusterIndex < clusterOfSamplesIndex->size; ++clusterIndex){
        iftGenericVector* cluster = iftVectorAt(iftGenericVector*,clusterOfSamplesIndex,clusterIndex);
        for (size_t sampleIndex = 0; sampleIndex < cluster->size; ++sampleIndex){
            int* list = (int*)cluster->data;
            int s = list[sampleIndex];
            if(filterCluster != 0 && workDataset->sample[s].group != filterCluster){
                continue;
            }
            if(filterClass != 0 && workDataset->sample[s].truelabel != filterClass){
                continue;
            }
            //if(workDataset->sample[s].group != )
            float dist = clusterDistance[clusterIndex][sampleIndex];
            if(dist > maxDist){
                maxDist = dist;
            }
        }
    }
    for (size_t clusterIndex = 0; clusterIndex < clusterOfSamplesIndex->size; ++clusterIndex){
        iftGenericVector* cluster = iftVectorAt(iftGenericVector*,clusterOfSamplesIndex,clusterIndex);
        for (size_t sampleIndex = 0; sampleIndex < cluster->size; ++sampleIndex){
            int* list = (int*)cluster->data;
            int s = list[sampleIndex];
            if(filterCluster != 0 && workDataset->sample[s].group != filterCluster){
                continue;
            }
            if(filterClass != 0 && workDataset->sample[s].truelabel != filterClass){
                continue;
            }
            float dist = clusterDistance[clusterIndex][sampleIndex];
            float index = dist/(maxDist+0.00001);
            float R,G,B;
            iftHeatColorMapping(index,&R,&G,&B);
            R = R*255;
            G = G*255;
            B = B*255;
            QColor cor = QColor(R,G,B,255);
            MyNode* node = hashSampleId2GraphNodeRef[workDataset->sample[s].id];
            node->fillColor = cor;
            node->isMarked = false;
        }
    }
    scene->update();
}

void MainWindow::paintNodeTextClusterFrontier(iftKnnGraph* graph){
    iftDataSet* dataset = graph->Z;
    int sampleIndex1;
    int sampleIndex2;
    for (int i = 0; i < graph->nnodes; ++i) {
        sampleIndex1 = graph->node[i].sample;
        MyNode* node = hashSampleId2GraphNodeRef[dataset->sample[sampleIndex1].id];
        node->isMarked = false;

        iftAdjSet* head = graph->node[i].adj;
        if (head != NULL){
            sampleIndex2 = graph->node[head->node].sample;

            if(dataset->sample[sampleIndex1].group != dataset->sample[sampleIndex2].group){
                if(graph->node[i].root != i){

                    MyNode* node = hashSampleId2GraphNodeRef[dataset->sample[sampleIndex1].id];
                    node->isMarked = true;
                    //                MyNode* node2 = hashSampleId2GraphNodeRef[dataset->sample[sampleIndex2].id];
                    //                node2->isMarked = true;
                }
            }
            //}
            //head = head->next;
        }
    }
    scene->update();
}

void MainWindow::paintNodeTextClusterHeatMap_geodesicDistance(iftKnnGraph* graph){
    float maxGeodesicDistance = -1;
    float minGeodesicDistance = 999999;

    for (int i = 0; i < graph->nnodes; ++i){
        int nodeIndex = graph->ordered_nodes[i];
        iftKnnNode iftNode = graph->node[nodeIndex];
        if(filterCluster != 0 && workDataset->sample[iftNode.sample].group != filterCluster){
            continue;
        }
        if(filterClass != 0 && workDataset->sample[iftNode.sample].truelabel != filterClass){
            continue;
        }
        if(maxGeodesicDistance < graph->pathval[nodeIndex]){
            maxGeodesicDistance = graph->pathval[nodeIndex];
            break;
        }
    }

    for (int i = graph->nnodes-1; i >= 0; --i){
        int nodeIndex = graph->ordered_nodes[i];
        iftKnnNode iftNode = graph->node[nodeIndex];
        if(filterCluster != 0 && workDataset->sample[iftNode.sample].group != filterCluster){
            continue;
        }
        if(filterClass != 0 && workDataset->sample[iftNode.sample].truelabel != filterClass){
            continue;
        }
        if(minGeodesicDistance > graph->pathval[nodeIndex]){
            minGeodesicDistance = graph->pathval[nodeIndex];
            break;
        }
    }

    for (int i = 0; i < graph->nnodes; ++i){
        int nodeIndex = graph->ordered_nodes[i];
        iftKnnNode iftNode = graph->node[nodeIndex];
        if(filterCluster != 0 && workDataset->sample[iftNode.sample].group != filterCluster){
            continue;
        }
        if(filterClass != 0 && workDataset->sample[iftNode.sample].truelabel != filterClass){
            continue;
        }
        int sampleIndex = iftNode.sample;
        float geodesicDistance = graph->pathval[nodeIndex];
        float normalizedDistance = (geodesicDistance-minGeodesicDistance)/(maxGeodesicDistance-minGeodesicDistance+0.00001);
        float R,G,B;
        iftHeatColorMapping(normalizedDistance,&R,&G,&B);
        R = R*255;
        G = G*255;
        B = B*255;
        QColor cor = QColor(R,G,B,255);
        MyNode* node = hashSampleId2GraphNodeRef[workDataset->sample[sampleIndex].id];
        if(node == NULL){
            continue;
        }
        node->fillColor = cor;
        node->isMarked = false;
        //node->update();
    }
    scene->update();
}

void MainWindow::paintNodeTextClusterHeatMap_geodesicDistanceNormalized(iftKnnGraph* graph){
    iftDataSet* dataset = graph->Z;
    float* maxGeodesicDistanceInCluster = (float*)iftAlloc(dataset->ngroups+10,sizeof(float));
    float* minGeodesicDistanceInCluster = (float*)iftAlloc(dataset->ngroups+10,sizeof(float));

    for (int c = 0; c < dataset->ngroups; ++c) {
        maxGeodesicDistanceInCluster[c] = -1;
        minGeodesicDistanceInCluster[c] = IFT_INFINITY_FLT;
    }

    for (int i = 0; i < graph->nnodes; ++i){
        int nodeIndex = graph->ordered_nodes[i];
        iftKnnNode iftNode = graph->node[nodeIndex];
        int sampleIndex =iftNode.sample;
        int group = dataset->sample[sampleIndex].group;

        if(maxGeodesicDistanceInCluster[group] < graph->pathval[nodeIndex]){
            maxGeodesicDistanceInCluster[group] = graph->pathval[nodeIndex];
        }
        if(minGeodesicDistanceInCluster[group] > graph->pathval[nodeIndex]){
            minGeodesicDistanceInCluster[group] = graph->pathval[nodeIndex];
        }
    }

    for (int i = 0; i < graph->nnodes; ++i){
        int nodeIndex = graph->ordered_nodes[i];
        iftKnnNode iftNode = graph->node[nodeIndex];
        int sampleIndex = iftNode.sample;
        int group = dataset->sample[sampleIndex].group;
        float maxValue = maxGeodesicDistanceInCluster[group];
        float minValue = minGeodesicDistanceInCluster[group];
        float geodesicDistance = graph->pathval[nodeIndex];
        float normalizedDistance = (geodesicDistance-minValue)/(maxValue-minValue+0.00001);
        float R,G,B;
        iftHeatColorMapping(normalizedDistance,&R,&G,&B);
        R = R*255;
        G = G*255;
        B = B*255;
        QColor cor = QColor(R,G,B,255);
        MyNode* node = hashSampleId2GraphNodeRef[dataset->sample[sampleIndex].id];
        if(node == NULL){
            continue;
        }
        node->fillColor = cor;
        node->isMarked = false;
        //node->update();
    }
    scene->update();
}

void MainWindow::paintNodeTextClusterHeatMap_inFrontier(iftKnnGraph* graph){
    iftDataSet* dataset = graph->Z;
    int sampleIndex1;
    int sampleIndex2;
    QVector<int> sampleIndices;
    QVector<float> sampleWeights;

    for (int i = 0; i < graph->nnodes; ++i) {
        int nodeIndex = graph->ordered_nodes[i];
        sampleIndex1 = graph->node[nodeIndex].sample;
        MyNode* node1 = hashSampleId2GraphNodeRef[dataset->sample[sampleIndex1].id];
        node1->isMarked = false;
        node1->fillColor = QColor(0,0,0,255);
        iftAdjSet* head = graph->node[nodeIndex].adj;
        if (head != NULL){
            sampleIndex2 = graph->node[head->node].sample;
            if(dataset->sample[sampleIndex1].group != dataset->sample[sampleIndex2].group){
                if(graph->node[nodeIndex].root != nodeIndex){
                    sampleIndices.push_back(sampleIndex1);
                    sampleWeights.push_back(dataset->sample[sampleIndex1].weight);
                }
            }
        }
    }
    float maxValue = IFT_INFINITY_FLT_NEG;
    float minValue = IFT_INFINITY_FLT;

    for (int i = 0; i < sampleWeights.size(); ++i) {
        if(sampleWeights.at(i) > maxValue){
            maxValue = sampleWeights.at(i);
        }
        if(sampleWeights.at(i) < minValue){
            minValue = sampleWeights.at(i);
        }
    }

    for (int i = 0; i < sampleIndices.size(); ++i) {
        int sampleIndex = sampleIndices.at(i);
        float sampleWeight = dataset->sample[sampleIndex].weight;
        float normalizedWeight = (sampleWeight-minValue)/(maxValue-minValue+0.00001);
        float R,G,B;
        iftHeatColorMapping(normalizedWeight,&R,&G,&B);
        R = R*255;
        G = G*255;
        B = B*255;
        QColor cor = QColor(R,G,B,255);
        MyNode* node = hashSampleId2GraphNodeRef[dataset->sample[sampleIndex].id];
        node->fillColor = cor;
        node->isMarked = false;
    }
    scene->update();
}


void MainWindow::paintNodeTextClusterSup(iftDataSet* dataset){
    if(hashSampleId2GraphNodeRef.size() <= 0){
        return;
    }
    int id;
    for (int sampleIndex = 0; sampleIndex < dataset->nsamples; ++sampleIndex) {
        id = dataset->sample[sampleIndex].id;
        MyNode* node = hashSampleId2GraphNodeRef[id];
        if(node == NULL){
            iftWarning("NUll node","paintNodePreictedClass");
            continue;
        }
        //node->drawOption = optionText;
        node->text = QString("%1").arg(dataset->sample[sampleIndex].group);
        if(dataset->sample[sampleIndex].isSupervised){
            QColor color = hashLabelId2LabelColor[dataset->sample[sampleIndex].label];
            node->fillColor = color;
        }else{
            node->fillColor = QColor(0,0,0,255);
        }
        node->isSupervised = dataset->sample[sampleIndex].isSupervised;
    }
    scene->update();
}

void MainWindow::paintNodeTextClusterAll(iftDataSet* dataset){
    if(hashSampleId2GraphNodeRef.size() <= 0){
        return;
    }
    int id;
    for (int sampleIndex = 0; sampleIndex < dataset->nsamples; ++sampleIndex) {
        id = dataset->sample[sampleIndex].id;
        MyNode* node = hashSampleId2GraphNodeRef[id];
        if(node == NULL){
            iftWarning("NUll node","paintNodePreictedClass");
            continue;
        }
        //node->drawOption = optionText;
        node->text = QString("%1").arg(dataset->sample[sampleIndex].group);
        QColor color = hashLabelId2LabelColor[dataset->sample[sampleIndex].label];
        node->fillColor = color;
        node->isSupervised = dataset->sample[sampleIndex].isSupervised;
    }
    scene->update();
}

void MainWindow::paintNodeTextClusterSupPropagated(iftDataSet* dataset){
    if(hashSampleId2GraphNodeRef.size() <= 0){
        return;
    }
    if(workDataset == NULL){
        return;
    }
    int id;
    for (int sampleIndex = 0; sampleIndex < dataset->nsamples; ++sampleIndex) {
        id = dataset->sample[sampleIndex].id;
        MyNode* node = hashSampleId2GraphNodeRef[id];
        if(node == NULL){
            iftWarning("NUll node","paintNodePreictedClass");
            continue;
        }
        //node->drawOption = optionText;
        node->text = QString("%1").arg(dataset->sample[sampleIndex].group);
        QColor color;
        if(dataset->sample[sampleIndex].isLabelPropagated == true || dataset->sample[sampleIndex].isSupervised){
            color = hashLabelId2LabelColor[dataset->sample[sampleIndex].label];
        }else{
            color = QColor(0,0,0,255);
        }
        node->fillColor = color;
    }
    scene->update();
}

void MainWindow::on_actionNext_State_triggered()
{
    go2NextAciveLeaningState();
}

void MainWindow::highlightNode(MyNode* node){
    node->radiusX = node->radiusX_highlight;
    node->radiusY = node->radiusY_highlight;
    node->borderColor_unselected = QColor(0,255,0,255);
    node->currentSupervising = true;
    node->Z = 1;
}

void MainWindow::unhighlighNode(MyNode* node){
    if(node == NULL){
        return;
    }
    node->radiusX = node->radiusX_unhighlight;
    node->radiusY = node->radiusY_unhighlight;
    node->borderColor_unselected = QColor(0,0,0,255);
    node->currentSupervising = false;
    node->Z = 0;
}

void MainWindow::updateNodeColor_ComboBox2Graph(const QString &comboBoxRow){
    int comboBoxRow_int = comboBoxRow.toInt();
    MyWidgetItem* myItem = ui->listWidget->itemRefs.at(comboBoxRow_int);
    int labelIndex = myItem->myComboBox->currentIndex();
    int sampleIndex = selectedSamplesIndicesVec.at(comboBoxRow_int);
    workDataset->sample[sampleIndex].truelabel = labelIndex;
    workDataset->sample[sampleIndex].label = labelIndex;
    int id = workDataset->sample[sampleIndex].id;
    MyNode* node = hashSampleId2GraphNodeRef[id];
    if(node != NULL){
        QColor color = hashLabelId2LabelColor[labelIndex];
        node->fillColor = color;
        node->update();
    }
}

void MainWindow::updateNodeColor_ComboBox2Graph2(const QString &comboBoxRow){
    int comboBoxRow_int = comboBoxRow.toInt();
    MyWidgetItem* myItem = mylist->itemRefs.at(comboBoxRow_int);
    int labelIndex = myItem->myComboBox->currentIndex();
    int sampleIndex = allSupervised.at(comboBoxRow_int);
    workDataset->sample[sampleIndex].truelabel = labelIndex;
    workDataset->sample[sampleIndex].label = labelIndex;
    int id = workDataset->sample[sampleIndex].id;
    MyNode* node = hashSampleId2GraphNodeRef[id];
    if(node != NULL){
        QColor color = hashLabelId2LabelColor[labelIndex];
        node->fillColor = color;
        node->update();
    }
    trainClassifier();
}

void MainWindow::setTrainAndTestSamples(){
    workDataset->ntrainsamples = 0;
    nSupervisedSamples = 0;
    nPropagatedLabelSamples = 0;
    for (int i = 0; i < workDataset->nsamples; ++i) {
        workDataset->sample[i].status = IFT_TEST;
        if(workDataset->sample[i].isSupervised ){
            workDataset->sample[i].status |= IFT_TRAIN;
            nSupervisedSamples++;
            workDataset->ntrainsamples++;
        }else if( workDataset->sample[i].isLabelPropagated){
            workDataset->sample[i].status |= IFT_TRAIN;
            nPropagatedLabelSamples++;
            workDataset->ntrainsamples++;
        }
        MyNode* node = hashSampleId2GraphNodeRef[workDataset->sample[i].id];
        workDataset->sample[i].numberTimesChecked = node->numberTimesChecked;
    }
}

void MainWindow::displaySelectedSamples(int start, int end){
    int id;
    int  sampleIndex;
    QString path2Image;
    QSignalMapper* signalMapper = new QSignalMapper(this);
    if(fileSet == NULL){
        for (int i = start; i < end; ++i) {
            sampleIndex = selectedSamplesIndicesVec.at(i);
            id = workDataset->sample[sampleIndex].id;
            MyWidgetItem* listItem = new MyWidgetItem(":/resource_images/Images/imageMissing.png",labelsNames,labelsColor,ui->listWidget);
            //            path2Image = "id: " + QString::number(id);
            //            listItem->setToolTip(path2Image);
            //            listItem->labelImage->setToolTip(path2Image);
            ui->listWidget->addMyItem(listItem);
            listItem->myComboBox->setCurrentIndex(workDataset->sample[sampleIndex].label);

            connect(listItem->myComboBox,SIGNAL(currentIndexChanged(int))
                    ,signalMapper,SLOT(map()));
            signalMapper->setMapping(listItem->myComboBox, QString("%1").arg(i));
            //}
        }
        connect(signalMapper, SIGNAL(mapped(const QString &)),
                this, SLOT(updateNodeColor_ComboBox2Graph(const QString &)));
    }
    else{
        QFileInfo *fileInfo = new QFileInfo();
        for (int i = start; i < end; ++i) {
            sampleIndex = selectedSamplesIndicesVec.at(i);
            id = workDataset->sample[sampleIndex].id;
            //listItem = hashSampleId2CurrentItemInList[id];
            //if(listItem == NULL){
            path2Image = fileSet->files[id]->path;
            QImage image(path2Image);
            MyWidgetItem* listItem ;
            if(!image.isNull()){
                listItem = new MyWidgetItem(path2Image,labelsNames,labelsColor,ui->listWidget);
                fileInfo->setFile(path2Image);
                //                listItem->setToolTip(fileInfo->fileName());
                //                listItem->labelImage->setToolTip(fileInfo->fileName());
            }else{
                listItem= new MyWidgetItem(":/resource_images/Images/imageMissing.png",labelsNames,labelsColor,ui->listWidget);
                fileInfo->setFile(path2Image);
                //                listItem->setToolTip(fileInfo->fileName());
                //                listItem->labelImage->setToolTip(fileInfo->fileName());
            }
            ui->listWidget->addMyItem(listItem);
            listItem->myComboBox->setCurrentIndex(workDataset->sample[sampleIndex].label);
            connect(listItem->myComboBox,SIGNAL(currentIndexChanged(int))
                    ,signalMapper,SLOT(map()));
            signalMapper->setMapping(listItem->myComboBox, QString("%1").arg(i));
            //}
        }
        delete fileInfo;
        connect(signalMapper, SIGNAL(mapped(const QString &)),
                this, SLOT(updateNodeColor_ComboBox2Graph(const QString &)));
    }

}

void MainWindow::undisplaySelectedSamplesF(){
    ui->listWidget->clear();
    ui->listWidget->itemRefs.clear();
}

void MainWindow::highlightSelectedSamples(){
    if(hashSampleId2GraphNodeRef.size() <= 0){
        return;
    }
    int sampleIndex;
    int id;
    for (int i = 0; i < selectedSamplesIndicesVec.size(); ++i) {
        sampleIndex = selectedSamplesIndicesVec.at(i);
        id = workDataset->sample[sampleIndex].id;
        MyNode* node = hashSampleId2GraphNodeRef[id];
        if(node == NULL){
            continue;
        }
        node->fillColor = hashLabelId2LabelColor[workDataset->sample[sampleIndex].label];
        highlightNode(node);
    }
    scene->update();
}

void MainWindow::unhighlightSelectedSamples(){
    int sampleIndex;
    int id;
    for (int i = 0; i < selectedSamplesIndicesVec.size(); ++i) {
        sampleIndex = selectedSamplesIndicesVec.at(i);
        id = workDataset->sample[sampleIndex].id;
        MyNode* node = hashSampleId2GraphNodeRef[id];
        if(node == NULL){
            continue;
        }
        unhighlighNode(node);
    }
    scene->update();
}

bool MainWindow::checkIfAllSamplesWereSupervised(){
    for (int i = 0; i < ui->listWidget->count(); ++i) {
        QListWidgetItem *itemObjectRef = ui->listWidget->item(i);
        MyWidgetItem* myItem = dynamic_cast<MyWidgetItem*>(ui->listWidget->itemWidget(itemObjectRef));
        if(myItem != NULL){
            if(myItem->myComboBox->currentIndex() == 0){
                QMessageBox::critical(this,"Error","There is a unsupervised sample");
                return false;
            }
        }else{
            QMessageBox::critical(this,"Error","There are NULL objects in list");
            return false;
        }
    }
    return true;
}

void MainWindow::findIconsForClassesFacebook(){
    if(ui->listWidget->count() <= 0 || ui->listWidget_2->count() <= 0){
        return;
    }
    if(fileSet == NULL){
        return;
    }
    bool allClasseshaveIcon = true;
    int trueLabel;
    if(ui->listWidget_2->allItemsHaveIcon == false){
        for (int i = 0; i < selectedSamplesIndicesVec.size(); ++i) {
            trueLabel = ui->listWidget->itemRefs.at(i)->myComboBox->currentIndex();
            if(trueLabel-1 >= 0){
                if(ui->listWidget_2->itemRefs.at(trueLabel-1)->hasRepresentativeImage == false){
                    QImage image(ui->listWidget->itemRefs.at(i)->path2Image);
                    if(!image.isNull()){
                        ui->listWidget_2->itemRefs.at(trueLabel-1)->updateImage(ui->listWidget->itemRefs.at(i)->path2Image);
                        ui->listWidget_2->itemRefs.at(trueLabel-1)->hasRepresentativeImage = true;
                        int sampleIndex = selectedSamplesIndicesVec.at(i);
                        ui->listWidget_2->itemRefs.at(trueLabel-1)->uniqueId = workDataset->sample[sampleIndex].id;
                        allClasseshaveIcon = false;
                    }
                }
            }
        }
        for (int var = 0; var < ui->listWidget_2->itemRefs.size(); ++var) {
            if(ui->listWidget_2->itemRefs.at(var)->hasRepresentativeImage == false){
                allClasseshaveIcon = false;
            }
        }
        ui->listWidget_2->allItemsHaveIcon = allClasseshaveIcon;
    }
}

void MainWindow::unfindIconsForClassesFacebook(){
    if(fileSet == NULL){
        return;
    }
    unsigned long uniqueId;
    for (int i = 0; i < selectedSamplesIndicesVec.size(); ++i) {
        int sampleIndex = selectedSamplesIndicesVec.at(i);
        uniqueId = workDataset->sample[sampleIndex].id;
        for (int j = 0; j < ui->listWidget_2->itemRefs.size(); ++j) {
            if(ui->listWidget_2->itemRefs.at(j)->uniqueId == uniqueId){
                ui->listWidget_2->allItemsHaveIcon = false;
                ui->listWidget_2->itemRefs.at(j)->updateImage(":/resource_images/Images/imageMissing.png");
                ui->listWidget_2->itemRefs.at(j)->hasRepresentativeImage = false;
                ui->listWidget_2->itemRefs.at(j)->path2Image = ":/resource_images/Images/imageMissing.png";
            }
        }
    }
}

void MainWindow::updateSamplesInDatasetToTrain(){
    int  sampleIndex;
    if(ui->listWidget->itemRefs.size() <= 0){
        QMessageBox::critical(this,"Error","listWidget is empty");
        return;
    }
    for (int i = 0; i < selectedSamplesIndicesVec.size(); ++i) {
        sampleIndex = selectedSamplesIndicesVec.at(i);
        workDataset->sample[sampleIndex].truelabel = ui->listWidget->itemRefs.at(i)->myComboBox->currentIndex();
        workDataset->sample[sampleIndex].label = ui->listWidget->itemRefs.at(i)->myComboBox->currentIndex();
        workDataset->sample[sampleIndex].status |= IFT_TRAIN;
        workDataset->sample[sampleIndex].isSupervised = true;
        MyNode* node = hashSampleId2GraphNodeRef[workDataset->sample[sampleIndex].id];
        node->isSupervised = true;
    }
}

void MainWindow::unupdateSamplesInDatasetToTrain(){
    int  sampleIndex;
    for (int i = 0; i < selectedSamplesIndicesVec.size(); ++i) {
        sampleIndex = selectedSamplesIndicesVec.at(i);
        workDataset->sample[sampleIndex].status &= 0b11111011;
        workDataset->sample[sampleIndex].isSupervised = false;
        MyNode* node = hashSampleId2GraphNodeRef[workDataset->sample[sampleIndex].id];
        //if(node != NULL){
        node->isSupervised = false;
        //}
    }
}

void MainWindow::addNewRegisterInHistoryVectors(){
    numberOfSupervisedSamplesHistory.push_back(numberOfSupervisedSamplesHistory.at(activeLearning->currentLearningCycle));
    learnCycleValues.push_back(activeLearning->currentLearningCycle+1);
}

void MainWindow::updateNumberOfSupervisedSamples(){
    numberOfSupervisedSamplesHistory[activeLearning->currentLearningCycle] += selectedSamplesIndicesVec.size();
}

void MainWindow::go2NextAciveLeaningState(){
    if(workDataset == NULL){
        QMessageBox::critical(this,"Error","Dataset is empty");
        return;
    }
    if(activeLearning == NULL){
        QMessageBox::critical(this,"Error","Active learner is empty");
        return;
    }
    if(selectedSamplesIndicesVec.size() == 0){
        QMessageBox::critical(this,"Error","There are not selected samples");
        return;
    }
    if(checkIfAllSamplesWereSupervised()){
        activeLearning->currentLearningCycle += 1;
        findIconsForClassesFacebook();
        updateSamplesInDatasetToTrain();
        undisplaySelectedSamplesF();
        unhighlightSelectedSamples();
        setTrainAndTestSamples();
        if(workDataset->ntrainsamples <= 0){
            QMessageBox::critical(this,"Error","There are not Train samples");
            return;
        }
        iftTrainGenericClassifier(activeLearning->classifier,workDataset);
        iftPredictGenericClassifier(activeLearning->classifier,workDataset);
        computeCrossValidationOnSupervisedSamples();
        if(developingMode == true){
            computeTrueLabelInfo();
        }
        updateTableWidget();

        if(activeLearning->currentLearningCycle == 0){
            numberOfSupervisedSamplesHistory[activeLearning->currentLearningCycle] = selectedSamplesIndicesVec.size();
        }else{
            int nSupervised = numberOfSupervisedSamplesHistory[activeLearning->currentLearningCycle-1] +selectedSamplesIndicesVec.size();
            numberOfSupervisedSamplesHistory[activeLearning->currentLearningCycle] = nSupervised;
        }


        learnCycleValues[activeLearning->currentLearningCycle] = activeLearning->currentLearningCycle;
        selectedSamplesIndicesVecHistory[activeLearning->currentLearningCycle] = selectedSamplesIndicesVec;
        selectedSamplesIndicesVec.clear();

        updatePlot();
        updateInfoArea();
        paintNode(currentNodeColorOption);
        QString datasetName = QDir::currentPath() + QString("/learningCycle_datasets/")
                + QString("dataset") + QString::number(activeLearning->currentLearningCycle)
                + QString("_") + QString::number(operationNumber[activeLearning->currentLearningCycle])+  QString(".zip");
        iftWriteDataSet(workDataset,datasetName.toLocal8Bit().constData());
        operationNumber[activeLearning->currentLearningCycle] += 1;
    }
}

void MainWindow::computeTrueLabelInfo(){
    trueLabelAcc = iftTruePositives(workDataset);
    trueLabelAccNorm = iftNormAccuracy(workDataset);
    trueLabelKappa = iftCohenKappaScore(workDataset);


    if(confusionMatrixCount != NULL){
        iftDestroyMatrix(&confusionMatrixCount);
    }
    confusionMatrixCount = iftComputeConfusionMatrix(workDataset,false);
    if(confusionMatrixNormalized != NULL){
        iftDestroyMatrix(&confusionMatrixNormalized);
    }
    confusionMatrixNormalized = iftComputeConfusionMatrix(workDataset,true);
    accTrueLabelVec[activeLearning->currentLearningCycle] = trueLabelAcc;
    accNormTrueLabelVec[activeLearning->currentLearningCycle] = trueLabelAccNorm;
    kappaTrueLabelVec[activeLearning->currentLearningCycle] = trueLabelKappa;
}

void MainWindow::unupdateNumberOfSupervisedSamples(){
    numberOfSupervisedSamplesHistory[activeLearning->currentLearningCycle] -= selectedSamplesIndicesVec.size();
}

void MainWindow::on_actionPrevious_State_triggered()
{
    go2PreviousAciveLeaningState();
}

void MainWindow::removeNewRegisterInHistoryVectors(){
    numberOfSupervisedSamplesHistory.pop_back();
    accuracy.pop_back();
    learnCycleValues.pop_back();
}

void MainWindow::ungetSamplesIndexRDS(){
    selectedSamplesIndicesVec.clear();
    selectedSamplesIndicesVecHistory.pop_back();
}

void MainWindow::computeCrossValidationOnSupervisedSamples(){
    int nValidSamples = workDataset->ntrainsamples;
    //    for (int i = 0; i < workDataset->nsamples; ++i) {
    //        if(workDataset->sample[i].isSupervised || workDataset->sample[i].isLabelPropagated){
    //            nValidSamples++;
    //        }
    //    }
    if(nValidSamples <= 1){
        return;
    }
    iftDataSet *crossValidationDataSet = iftCreateDataSet(nValidSamples,workDataset->nfeats);
    crossValidationDataSet->nclasses = workDataset->nclasses;
    int k = 0;
    for (int i = 0; i < workDataset->nsamples; ++i) {
        if(workDataset->sample[i].isSupervised || workDataset->sample[i].isLabelPropagated){
            iftCopySample(&(workDataset->sample[i]),
                          &(crossValidationDataSet->sample[k]),
                          workDataset->nfeats,
                          false);
            k++;
        }
    }
    int times = 5;
    int nsplits = 2;
    iftSampler* kfold = iftNKFold(nValidSamples, times,nsplits);
    float *Acc  = iftAllocFloatArray(kfold->niters);
    float *AccNorms  = iftAllocFloatArray(kfold->niters);
    float *kappas  = iftAllocFloatArray(kfold->niters);
    iftGenericVector* vectorMatrix = iftCreateGenericVector(kfold->niters,sizeof(iftMatrix*));
    iftGenericClassifier* classifier = iftCreateGenericClassifier(CLASSIFIER_OPF_SUP);

    for (int i=0; i < kfold->niters; i++) {
        iftDataSetSampling(crossValidationDataSet, kfold, i);
        iftTrainGenericClassifier(classifier,crossValidationDataSet);
        iftPredictGenericClassifier(classifier,crossValidationDataSet);
        iftMatrix* confusionMatrix = iftComputeConfusionMatrix(crossValidationDataSet,true);
        //iftDestroyMatrix(&confusionMatrix);
        iftGenericVectorPushBack(iftMatrix*, vectorMatrix,confusionMatrix);
        Acc[i] = iftTruePositives(crossValidationDataSet);
        AccNorms[i] = iftNormAccuracy(crossValidationDataSet);
        kappas[i] = iftCohenKappaScore(crossValidationDataSet);

    }
    meanAcc = iftMean(Acc,kfold->niters);
    stdevAcc = iftStddevFloatArray(Acc,kfold->niters);
    meanAccNorm = iftMean(AccNorms,kfold->niters);
    stdevAccNorm = iftStddevFloatArray(AccNorms,kfold->niters);
    meanKappa = iftMean(kappas,kfold->niters);
    stdevKappa = iftStddevFloatArray(kappas,kfold->niters);

    accSmallDataset[activeLearning->currentLearningCycle] = meanAcc;
    accStdSmallDataset[activeLearning->currentLearningCycle] = stdevAcc;
    accNormSmallDataset[activeLearning->currentLearningCycle] = meanAccNorm;
    accNormStdSmallDataset[activeLearning->currentLearningCycle] = stdevAccNorm;
    kappaSmallDataset[activeLearning->currentLearningCycle] = meanKappa;
    kappaStdSmallDataset[activeLearning->currentLearningCycle] = stdevKappa;

    if(confusionMatrixSmallDatasetMean == NULL){
        confusionMatrixSmallDatasetMean = iftCreateMatrix(workDataset->nclasses,workDataset->nclasses);
    }else{
        iftSetMatrix(confusionMatrixSmallDatasetMean,0);
    }

    if(confusionMatrixSmallDatasetStandardDeviation == NULL){
        confusionMatrixSmallDatasetStandardDeviation = iftCreateMatrix(workDataset->nclasses,workDataset->nclasses);
    }else{
        iftSetMatrix(confusionMatrixSmallDatasetStandardDeviation,0);
    }

    for (size_t i = 0; i < vectorMatrix->size; ++i) {
        iftMatrix* confusionMatrix = iftVectorAt(iftMatrix*,vectorMatrix,i);
        iftMatricesAdditionPointWiseInPlace(confusionMatrix,
                                            confusionMatrixSmallDatasetMean,
                                            &(confusionMatrixSmallDatasetMean));
    }
    iftMultMatrixByScalar(confusionMatrixSmallDatasetMean,(1/(kfold->niters+0.0000001)));

    for (size_t i = 0; i < vectorMatrix->size; ++i) {
        iftMatrix* confusionMatrix = iftVectorAt(iftMatrix*,vectorMatrix,i);
        for (int row = 0; row < confusionMatrixSmallDatasetStandardDeviation->nrows; ++row) {
            for (int col = 0; col < confusionMatrixSmallDatasetStandardDeviation->ncols; ++col) {
                float value = iftMatrixElem(confusionMatrix,col,row) - iftMatrixElem(confusionMatrixSmallDatasetMean,col,row);
                value = value*value;
                iftMatrixElem(confusionMatrixSmallDatasetStandardDeviation,col,row) += value;
            }
        }
    }
    iftMultMatrixByScalar(confusionMatrixSmallDatasetStandardDeviation,(1/(kfold->niters+0.0000001)));
    for (int row = 0; row < confusionMatrixSmallDatasetStandardDeviation->nrows; ++row) {
        for (int col = 0; col < confusionMatrixSmallDatasetStandardDeviation->ncols; ++col) {
            iftMatrixElem(confusionMatrixSmallDatasetStandardDeviation,col,row) =
                    sqrt(iftMatrixElem(confusionMatrixSmallDatasetStandardDeviation,col,row));
        }
    }
    for (size_t i = 0; i < vectorMatrix->size; ++i) {
        iftMatrix* confusionMatrix = iftVectorAt(iftMatrix*,vectorMatrix,i);
        iftDestroyMatrix(&confusionMatrix);
    }

    iftDestroyGenericClassifier(&classifier);
    iftDestroySampler(&kfold);
    iftDestroyDataSet(&crossValidationDataSet);
}


void MainWindow::go2PreviousAciveLeaningState(){
    if(workDataset == NULL){
        QMessageBox::critical(this,"Error","Dataset is empty");
        return;
    }
    if(activeLearning == NULL){
        QMessageBox::critical(this,"Error","Active learner is empty");
        return;
    }
    if(activeLearning->currentLearningCycle == 0){
        QMessageBox::critical(this,"Error","current learning cycle is 0");
        return;
    }

    ui->listWidget->clear();
    ui->listWidget->itemRefs.clear();
    for (int i = 0; i < selectedSamplesIndicesVec.size(); ++i) {
        int sampleIndex = selectedSamplesIndicesVec.at(i);
        MyNode* node = hashSampleId2GraphNodeRef[workDataset->sample[sampleIndex].id];
        unhighlighNode(node);
    }
    //activeLearning->currentLearningCycle -=1;


    selectedSamplesIndicesVec = selectedSamplesIndicesVecHistory.at(activeLearning->currentLearningCycle);

    for (int i = 0; i < selectedSamplesIndicesVec.size(); ++i) {
        int sampleIndex = selectedSamplesIndicesVec.at(i);
        int id = workDataset->sample[sampleIndex].id;
        hashSampleId2CurrentSelected.insert(id,false);
    }


    //selectedSamplesIndicesVec = selectedSamplesIndicesVecHistory.takeAt(activeLearning->currentLearningCycle-1);
    activeLearning->currentLearningCycle -= 1;
    unfindIconsForClassesFacebook();
    unupdateSamplesInDatasetToTrain();
    //displaySelectedSamples(0,selectedSamplesIndicesVec.size());
    //highlightSelectedSamples();
    selectedSamplesIndicesVec.clear();
    updatePlot();
    updateInfoArea();
    paintNode(currentNodeColorOption);
}

void MainWindow::on_actionMetrics_triggered()
{
    updatePlot();
    // add a color scale:
    // rescale the key (x) and value (y) axes so the whole color map is visible:

    plot->show();
}

void MainWindow::updatePlot(){
    if(activeLearning == NULL){
        return;
    }
    QVector<double>X;
    QVector<double>Y;
    size_t i;
    for (i = 0; i < activeLearning->currentLearningCycle; ++i) {
        X.push_back(i);
        Y.push_back(accuracy.at(i));
    }
    plot->xAxis->setRange(0.0,i);
    plot->graph(0)->setData(X,Y,true);
    //plot->rescaleAxes();
    plot->replot();
}


void MainWindow::on_pushButton_clicked()
{
    iftDataSet* dataset = iftCopyDataSet(workDataset,false);
    setStatusSupervisedandUnsupervisedSamples(dataset,IFT_TRAIN, IFT_TEST);
    if(dataset->ntrainsamples <= 0){
        QMessageBox::warning(this,"warning","there are not supervised samples");
        return;
    }
    iftTrainGenericClassifier(activeLearning->classifier,dataset);
    iftSetStatus(dataset,ALL);
    //iftPredictGenericClassifier(activeLearning->classifier,dataset);
    float accuracyValue = iftTruePositives(dataset);
    accuracy[activeLearning->currentLearningCycle] = accuracyValue;
    qDebug() << accuracyValue;
    updatePlot();
    updateInfoArea();
    free(dataset);
}

void MainWindow::setStatusSupervisedandUnsupervisedSamples(iftDataSet* dataset,iftSampleStatus statusSupervised,iftSampleStatus statusUnsupervised){
    dataset->ntrainsamples = 0;
    for (int i = 0; i < dataset->nsamples; ++i) {
        if(dataset->sample[i].isSupervised && dataset->sample[i].truelabel > 0){
            dataset->sample[i].status = statusSupervised;
            dataset->ntrainsamples += 1;
        }else{
            dataset->sample[i].status = statusUnsupervised;
        }
    }
}


void MainWindow::updateDrawingOptionForNodes(QString option){
    if(option == "Drawing"){
        for (int i= 0; i < graphNodes.size(); ++i) {
            graphNodes.at(i)->drawOption = optionDrawing;
        }
    }else if(option == "Text"){
        for (int i= 0; i < graphNodes.size(); ++i) {
            graphNodes.at(i)->drawOption = optionText;
        }
    }else if(option == "Image"){
        for (int i= 0; i < graphNodes.size(); ++i) {
            graphNodes.at(i)->drawOption = optionImage;
        }
    }
    scene->update();
}


void MainWindow::on_pushButton_2_clicked()
{
    selectSamplesManually();
}

void MainWindow::on_pushButton_3_clicked()
{
    selectSamplesAutomatically();
}

void MainWindow::selectSamplesManually()
{
    if(graphNodes.size() <= 0){
        QMessageBox::critical(this,"Error","There is not graph");
        return;
    }
    int id;
    int row;
    int lastIndex = selectedSamplesIndicesVec.size();
    for (int i = 0; i < graphNodes.size(); ++i) {
        if(graphNodes.at(i)->isSelected()){
            id = hashGraphNodeRef2SampleId[graphNodes.at(i)];
            bool selected = hashSampleId2CurrentSelected[id];
            if(selected == false){
                row = hashSampleId2DatasetRow[id];
                hashSampleId2CurrentSelected.insert(id,true);
                selectedSamplesIndicesVec.push_back(row);
            }
        }
    }
    displaySelectedSamples(lastIndex,selectedSamplesIndicesVec.size());
    highlightSelectedSamples();
    updateInfoArea();
    //gambiarra para resolver um bug do Qt
    if(marginOscilator_forBugCorrection == 1){
        marginOscilator_forBugCorrection = -1;
    }else{
        marginOscilator_forBugCorrection = 1;
    }

    ui->listWidget->setGeometry(ui->listWidget->geometry().x(),ui->listWidget->geometry().y(),
                                ui->listWidget->geometry().width(),ui->listWidget->geometry().height()+marginOscilator_forBugCorrection);
}

void MainWindow::selectSamplesAutomatically(){
    iftIntArray* selectedIndices = iftSelectSamplesGenericSelector(selector);
    int* indices = selectedIndices->val;
    int sampleIndex;
    int id;
    iftDataSet* dataset = workDataset;
    int lastIndex = selectedSamplesIndicesVec.size();
    for (size_t i = 0; i < selectedIndices->n; ++i) {
        sampleIndex = indices[i];
        id = dataset->sample[sampleIndex].id;
        bool selected = hashSampleId2CurrentSelected[id];
        if(selected == false){
            hashSampleId2CurrentSelected.insert(id,true);
            selectedSamplesIndicesVec.push_back(sampleIndex);
        }
    }
    displaySelectedSamples(lastIndex,selectedSamplesIndicesVec.size());
    highlightSelectedSamples();
    printf("**selected samples**\n");
    //iftPrintSelectedSamplesInfoRDS(selectorRDS);
    updateInfoArea();
    //    ui->listWidget->repaint();
    //    ui->listWidget_2->repaint();
    //    ui->listWidget->update();
    //    ui->listWidget_2->update();
    //gambiarra para resolver um bug do Qt
    if(marginOscilator_forBugCorrection == 1){
        marginOscilator_forBugCorrection = -1;
    }else{
        marginOscilator_forBugCorrection = 1;
    }

    ui->listWidget->setGeometry(ui->listWidget->geometry().x(),ui->listWidget->geometry().y(),
                                ui->listWidget->geometry().width(),ui->listWidget->geometry().height()+marginOscilator_forBugCorrection);
}

void MainWindow::updateDisplayOfSelectedSamples_removed(QVector<int> indices){
    if(indices.size() == 0){
        return;
    }
    int sampleIndex;
    int id;
    for (int i = indices.size()-1; i >=  0; --i) {
        //int index = indices.at(i);
        sampleIndex = selectedSamplesIndicesVec.at(i);
        selectedSamplesIndicesVec.remove(i);
        id = workDataset->sample[sampleIndex].id;
        MyNode* node = hashSampleId2GraphNodeRef[id];
        if(node != NULL){
            unhighlighNode(node);
        }
        hashSampleId2CurrentSelected.remove(id);
    }
    scene->update();
    updateInfoArea();
}


void MainWindow::setDrawEdges(const QString &arg1){
    if(arg1 == "None"){
        drawNoneEdges(workDataset);
    }else if(arg1 == "Predecessor_path"){
        if(selectorRDS == NULL){
            return;
        }
        drawPredecessorEdges(selectorRDS->opfClusteringGrapgh);
    }else if(arg1 == "k-nearest"){
        if(selectorRDS == NULL){
            return;
        }
        drawNeighbourEdges(selectorRDS->opfClusteringGrapgh);
    }
}

void MainWindow::drawNoneEdges(iftDataSet* dataset){
    graphEdges.clear();
    for (int i = 0; i < hashSampleId2GraphNodeRef.size(); ++i) {
        MyNode* node = hashSampleId2GraphNodeRef[dataset->sample[i].id];
        node->edgeMode = EdgeMode_None;
    }
    scene->update();
}

void MainWindow::drawPredecessorEdges(iftKnnGraph* graph){
    if(graph == NULL){
        return;
    }
    graphEdges.clear();
    iftDataSet* dataset = graph->Z;
    for (int i = 0; i < graph->nnodes; ++i) {
        iftKnnNode node = graph->node[graph->ordered_nodes[i]];
        int sample = node.sample;
        MyNode* nodeSource = hashSampleId2GraphNodeRef[dataset->sample[sample].id];
        iftKnnNode predecessorNode;
        int samplePredecessor;
        MyNode* nodeDestination;
        if(nodeSource != NULL){
            nodeSource->edgeMode = EdgeMode_Predecessor;
        }
        if(node.pred >= 0){
            predecessorNode =  graph->node[node.pred];
            samplePredecessor = predecessorNode.sample;
            nodeDestination = hashSampleId2GraphNodeRef[dataset->sample[samplePredecessor].id];
            if(nodeDestination != NULL){
                nodeDestination->edgeMode = EdgeMode_Predecessor;
                MyEdge *edge = new MyEdge(nodeSource,nodeDestination);
                edge->setVisible(false);
                edge->lineWidth = 0.5;
                graphEdges.push_back(edge);
                scene->addItem(edge);
            }
        }
    }
    scene->update();
}

void MainWindow::drawNeighbourEdges(iftKnnGraph* graph){
    if(graph == NULL){
        return;
    }
    graphEdges.clear();
    iftDataSet* dataset = graph->Z;
    for (int i = 0; i < graph->nnodes; ++i) {
        iftKnnNode node = graph->node[graph->ordered_nodes[i]];
        iftAdjSet *adjset = node.adj;
        while(adjset != NULL){
            iftKnnNode nodeDest = graph->node[adjset->node];
            MyNode* nodeSource = hashSampleId2GraphNodeRef[dataset->sample[node.sample].id];
            MyNode* nodeDestination = hashSampleId2GraphNodeRef[dataset->sample[nodeDest.sample].id];
            if(nodeSource != NULL && nodeDestination != NULL){
                nodeDestination->edgeMode = EdgeMode_Neighbours;
                MyEdge *edge = new MyEdge(nodeSource,nodeDestination);
                edge->arrowSize = 0;
                edge->setVisible(false);
                graphEdges.push_back(edge);
                scene->addItem(edge);
            }
            adjset = adjset->next;
        }
    }
    scene->update();
}

void MainWindow::on_spinBox_valueChanged(int arg1)
{
    iftSelectorRDS* rds = (iftSelectorRDS*)activeLearning->selector->selector;
    rds->numberOfSelectedSamplesPerCluster = arg1;
}


void MainWindow::setDrawNodeBinary(const QString &arg1){
    if(arg1 == "None"){
        drawNodeBinaryNone(workDataset);
    }else if(arg1 == "Cluster_frontier"){
        if(selectorRDS == NULL){
            return;
        }
        drawNodeBinaryFrontierCluster(selectorRDS->opfClusteringGrapgh);
    }else if(arg1 == "Root_mismatch"){
        if(selectorRDS == NULL){
            return;
        }
        drawNodeBinaryRootMismatch(workDataset,selectorRDS->clusterOfSamplesIndex);
    }
}

void MainWindow::drawNodeBinaryNone(iftDataSet* dataset){
    if(dataset == NULL || hashSampleId2GraphNodeRef.size() <= 0){
        return;
    }
    int id;
    for (int i = 0; i < dataset->nsamples; ++i) {
        id = dataset->sample[i].id;
        MyNode* node = hashSampleId2GraphNodeRef[id];
        if(node != NULL){
            node->isMarked = false;
            node->setVisible(true);
        }
    }
    scene->update();
}

void MainWindow::drawNodeBinaryFrontierCluster(iftKnnGraph* graph){
    iftDataSet* dataset = graph->Z;
    int sampleIndex1;
    int sampleIndex2;
    for (int i = 0; i < graph->nnodes; ++i) {
        sampleIndex1 = graph->node[i].sample;
        MyNode* node = hashSampleId2GraphNodeRef[dataset->sample[sampleIndex1].id];
        //node->setVisible(false);
        node->isMarked = false;
        iftAdjSet* head = graph->node[i].adj;
        if (head != NULL){
            sampleIndex2 = graph->node[head->node].sample;
            if(dataset->sample[sampleIndex1].group != dataset->sample[sampleIndex2].group){
                if(graph->node[i].root != i){
                    MyNode* node = hashSampleId2GraphNodeRef[dataset->sample[sampleIndex1].id];
                    node->isMarked = true;
                    //node->setVisible(true);
                }
            }
        }
    }
    scene->update();
}

void MainWindow::drawNodeBinaryRootMismatch(iftDataSet* dataset,iftGenericVector* clusterOfSamplesIndex){
    for (size_t clusterIndex = 0; clusterIndex < clusterOfSamplesIndex->size; ++clusterIndex){
        iftGenericVector* cluster = iftVectorAt(iftGenericVector*,clusterOfSamplesIndex,clusterIndex);
        int* list = (int*)cluster->data;
        int rootSampleIndex;
        int rootLabel;
        for (size_t sampleIndex = 0; sampleIndex < cluster->size; ++sampleIndex){
            int s = list[sampleIndex];
            MyNode* node = hashSampleId2GraphNodeRef[dataset->sample[s].id];
            node->isMarked = false;
            if(sampleIndex == 0){
                rootSampleIndex = s;
                rootLabel = dataset->sample[rootSampleIndex].label;
                continue;
            }
            if(filterCluster != 0 && dataset->sample[s].group != filterCluster){
                continue;
            }
            if(filterClass != 0 && dataset->sample[s].truelabel != filterClass){
                continue;
            }
            int sampleLabel = dataset->sample[s].label;
            if(sampleLabel != rootLabel){
                MyNode* node = hashSampleId2GraphNodeRef[dataset->sample[s].id];
                node->isMarked = true;
            }
        }
    }
    scene->update();
}

void MainWindow::on_pushButton_4_clicked()
{
    openWindowSupervisedSamples();
}

void MainWindow::openWindowSupervisedSamples(){
    if(workDataset == NULL){
        QMessageBox::warning(this,"warning","Dataset is NULL");
        return;
    }

    int id;
    int  sampleIndex;
    QString path2Image;
    QSignalMapper* signalMapper = new QSignalMapper(this);
    allSupervised.clear();
    //    if(mylist != NULL){
    //        delete mylist;
    //        mylist = NULL;
    //    }
    mylist = new MyListWidget();
    if(fileSet == NULL){
        int k=0;
        for (int i = 0; i < workDataset->nsamples; ++i) {

            //for (int i = 0; i < vec.size(); ++i) {
            if(workDataset->sample[i].isSupervised || workDataset->sample[i].isLabelPropagated){
                sampleIndex = i;
                allSupervised.push_back(sampleIndex);
                id = workDataset->sample[sampleIndex].id;
                MyWidgetItem* listItem = new MyWidgetItem(":/resource_images/Images/imageMissing.png",labelsNames,labelsColor,mylist);
                path2Image = "id: " + QString::number(id);
                listItem->setToolTip(path2Image);
                listItem->labelImage->setToolTip(path2Image);
                mylist->addMyItem(listItem);
                listItem->myComboBox->setCurrentIndex(workDataset->sample[sampleIndex].label);
                connect(listItem->myComboBox,SIGNAL(currentIndexChanged(int))
                        ,signalMapper,SLOT(map()));
                signalMapper->setMapping(listItem->myComboBox, QString("%1").arg(k));
                k++;
            }
        }
        connect(signalMapper, SIGNAL(mapped(const QString &)),
                this, SLOT(updateNodeColor_ComboBox2Graph2(const QString &)));
    }
    else{
        int k=0;
        for (int i = 0; i < workDataset->nsamples; ++i) {

            if(workDataset->sample[i].isSupervised || workDataset->sample[i].isLabelPropagated){
                sampleIndex = i;
                allSupervised.push_back(sampleIndex);
                id = workDataset->sample[sampleIndex].id;
                path2Image = fileSet->files[id]->path;
                QImage image(path2Image);
                MyWidgetItem* listItem ;
                if(!image.isNull()){
                    listItem = new MyWidgetItem(path2Image,labelsNames,labelsColor,mylist);
                }else{
                    listItem = new MyWidgetItem(":/resource_images/Images/imageMissing.png",labelsNames,labelsColor,mylist);
                    listItem->setToolTip(path2Image);
                    listItem->labelImage->setToolTip(path2Image);
                }
                mylist->addMyItem(listItem);
                listItem->myComboBox->setCurrentIndex(workDataset->sample[sampleIndex].label);
                connect(listItem->myComboBox,SIGNAL(currentIndexChanged(int))
                        ,signalMapper,SLOT(map()));
                signalMapper->setMapping(listItem->myComboBox, QString("%1").arg(k));
                k++;
            }
        }
        connect(signalMapper, SIGNAL(mapped(const QString &)),
                this, SLOT(updateNodeColor_ComboBox2Graph2(const QString &)));

    }
    mylist->setGeometry(0,0,500,400);
    mylist->show();
    mylist->setAttribute( Qt::WA_DeleteOnClose );
    mylist->setWindowTitle("Supervised Samples");
    connect( mylist, SIGNAL(destroyed(QObject*)), this, SLOT(mylistDestroyed(QObject*)) );
}

void MainWindow::mylistDestroyed(QObject*){
    setTrainAndTestSamples();
    iftTrainGenericClassifier(activeLearning->classifier,workDataset);
    iftPredictGenericClassifier(activeLearning->classifier,workDataset);
    iftMatrix* confusionMatrix = iftComputeConfusionMatrix(workDataset,false);
    iftDestroyMatrix(&confusionMatrix);
    iftMatrix* confusionMatrixN = iftComputeConfusionMatrix(workDataset,true);
    iftDestroyMatrix(&confusionMatrixN);

    float accuracyValue = iftTruePositives(workDataset);
    accuracy[accuracy.size()-1] = accuracyValue;
    updateInfoArea();
}

void MainWindow::on_actionactionLoadLabels_triggered()
{
    loadTextlabels();
}

bool orderPairByAlphabeticalOrder(const QPair<int,QString> &v1, const QPair<int,QString> &v2)
{
    return v1.second < v2.second;
}

bool orderPairByTrueLabelOrder(const QPair<int,QString> &v1, const QPair<int,QString> &v2)
{
    return v1.first < v2.first;
}

void MainWindow::loadTextlabels(){
    QString fileName_text_categories = QFileDialog::getOpenFileName(this, tr("Open Text file"), "~/",
                                                                    tr("Text file (*.*);;"));
    if(!fileName_text_categories.isNull() && !fileName_text_categories.isEmpty()){
        QFile inputFile(fileName_text_categories);
        if (inputFile.open(QIODevice::ReadOnly)){
            QList< QPair< int, QString > >listPairTrueLabelIdAndTrueLabelName;
            QTextStream in(&inputFile);
            int i=0;

            while (!in.atEnd())
            {
                QStringList stringList= in.readLine().split(" ");
                QPair< int, QString > pair;
                QString numberString = stringList.at(0);
                //QString numberString = in.readLine();
                pair.first = numberString.toInt();
                pair.second = stringList.at(1);
                listPairTrueLabelIdAndTrueLabelName.append(pair);
                i++;
            }


            //alphabetical order
            std::sort( listPairTrueLabelIdAndTrueLabelName.begin(),
                       listPairTrueLabelIdAndTrueLabelName.end(), orderPairByTrueLabelOrder);

            for (int var = 0; var <  listPairTrueLabelIdAndTrueLabelName.size(); ++var) {
                hashLabelId2LabelName.insert(listPairTrueLabelIdAndTrueLabelName.at(var).first,
                                             listPairTrueLabelIdAndTrueLabelName.at(var).second);
                qDebug() << listPairTrueLabelIdAndTrueLabelName.at(var).first << hashLabelId2LabelName[listPairTrueLabelIdAndTrueLabelName.at(var).first];
            }
            labelsNames.clear();
            for (int i = 0; i < hashLabelId2LabelName.size(); ++i) {
                labelsNames << hashLabelId2LabelName[i];
            }
            if(listPairTrueLabelIdAndTrueLabelName.size() > hashLabelId2LabelColor.size()){
                int nclasses = listPairTrueLabelIdAndTrueLabelName.size();
                QColor color;
                double h = 0.0;
                double s = 1.0;
                double v = 1.0;
                double step = 1.0/nclasses;
                hashLabelId2LabelColor.insert(0,QColor(0,0,0,200));
                for (int i = 0; i < nclasses; ++i) {
                    color.setHsvF(h,s,v);
                    h += step;
                    color.setAlpha(200);
                    hashLabelId2LabelColor.insert(i+1,QColor(color));
                }
                labelsColor.clear();
                for (int i = 0; i < hashLabelId2LabelColor.size(); ++i) {
                    labelsColor << hashLabelId2LabelColor[i];
                }
            }
            inputFile.close();
            windowSceneOptions->fillPropagationComboBox(hashLabelId2LabelName,hashLabelId2LabelColor);
            windowSceneOptions->fillClassFilterComboBox(hashLabelId2LabelName,hashLabelId2LabelColor);

            QTableWidget* table = ui->tableWidget;
            QStringList headers;
            for (int i = 1; i < hashLabelId2LabelName.size(); ++i) {
                headers << hashLabelId2LabelName[i];
            }
            //update confusion matrix headers
            table->setHorizontalHeaderLabels(headers);
            table->setVerticalHeaderLabels(headers);

            //update facebook samples names
//            MyListWidget* facebookList = ui->listWidget_2;
//            for (int i = 0; i < facebookList->itemRefs.size(); ++i) {
//                QListWidgetItem* item = facebookList->item(i);
//                MyWidgetItem* myItem = dynamic_cast<MyWidgetItem*>(item);
//                myItem->myComboBox->setCurrentText("dqweiidgy");
//            }


        }else{
            QMessageBox::critical(0,"Error","Could not open file %s",fileName_text_categories);
        }
    }
}

void MainWindow::mountConfusionMatrixWidget(){
    //ui->tableWidget = new QTableWidget(workDataset->nclasses,workDataset->nclasses,this);
    QTableWidget* table = ui->tableWidget;
    table->clear();
    table->setRowCount(workDataset->nclasses);
    table->setColumnCount(workDataset->nclasses);
    QStringList headers;

    for (int i = 1; i < hashLabelId2LabelName.size(); ++i) {
        headers << hashLabelId2LabelName[i];
    }
    table->setHorizontalHeaderLabels(headers);
    table->setVerticalHeaderLabels(headers);
    //    QLabel *label = new QLabel("miau");
    //    QHBoxLayout* layout = new QHBoxLayout(label);
    //    layout->addWidget(label);
    //    layout->setAlignment( Qt::AlignCenter );
    //    layout->setSpacing(0);
    //    label->setLayout(layout);
    //    table->setCellWidget(0,0,label);

    for (int row = 0; row < workDataset->nclasses; ++row) {
        for (int col = 0; col < workDataset->nclasses; ++col) {
            QLabel *label = new QLabel("0.0000");
            QHBoxLayout* layout = new QHBoxLayout();
            layout->setAlignment( Qt::AlignCenter );
            layout->setSpacing(0);
            label->setLayout(layout);
            table->setCellWidget(row,col,label);
        }
    }
}

void MainWindow::updateTableWidget(){
    QTableWidget* table = ui->tableWidget;
    float acc;
    QString text;
    iftMatrix* usedMatrix;
    if(seeTrueLabelInfo == true){
        usedMatrix = confusionMatrixNormalized;
    }else{
        usedMatrix = confusionMatrixSmallDatasetMean;
    }

    for (int row = 0; row < workDataset->nclasses; ++row) {
        for (int col = 0; col < workDataset->nclasses; ++col) {
            acc = iftMatrixElem(usedMatrix,col,row);
            QWidget * tableItem= table->cellWidget(row,col);
            QLabel* label = dynamic_cast<QLabel*>(tableItem);
            text = QString::number(acc,10,4);
            label->setText(text);
        }
    }
}

void MainWindow::on_pushButton_5_clicked()
{
    reprojectVisualization();
}

void MainWindow::reprojectVisualization(){
    if(workDataset == NULL){
        return;
    }
    if(graphNodes.size() <= 0){
        return;
    }
    if(scene == NULL){
        return;
    }
    float x,y;
    float rx = pointDefaultRx;
    float ry = pointDefaultRy;
    float scalinFactor_width = (myGraphicWidgetProjection->size().width()-(rx*4));
    float scalinFactor_height = (myGraphicWidgetProjection->size().height()-(ry*4));
    float factor;
    if(scalinFactor_height < scalinFactor_width){
        sceneScalingFactor = scalinFactor_height*2;
        factor = sceneScalingFactor;
        sceneScalingFactor = scalinFactor_height*2 + ry*4;
        //factor = scalinFactor_height*2;
    }else{
        //factor = scalinFactor_width*2;
        sceneScalingFactor = scalinFactor_width*2;
        factor = sceneScalingFactor;
        sceneScalingFactor = scalinFactor_width*2 + rx*4;
    }

    for (int i = 0; i < graphNodes.size(); ++i) {
        x = (workDataset->projection->val[i*2])*factor  + rx*2;
        y = (workDataset->projection->val[i*2 + 1])*factor + ry*2;
        MyNode* node = graphNodes.at(i);
        node->setPos(qreal(x),qreal(y));
    }
    ui->graphicsView->scene()->setSceneRect(scene->sceneRect().x(),scene->sceneRect().y(),factor+(rx*4), factor+(ry*4));

    //scene->invalidate(QRectF(0,0,100,100),QGraphicsScene::BackgroundLayer);
    //myGraphicWidgetProjection->drawBackground();
    scene->update();
}

void MainWindow::on_actionSave_Dataset_triggered()
{
    saveWorkDataset();
}

void MainWindow::saveWorkDataset(){
    if(workDataset == NULL){
        QMessageBox::critical(this,"Error","Dataset is empty");
        return;
    }
    QString filename = QFileDialog::getSaveFileName(this,"Save file","",".zip");
    filename.append(".zip");
    if(filename.isEmpty()){
        return;
    }
    iftWriteDataSet(workDataset,filename.toLatin1().data());
}

MyThread::MyThread(MainWindow* window, ThreadTaskType taskType)
{
    this->mainwindow = window;
    this->taskType = taskType;
}

void MyThread::run(){

    //m_NotificationTimer.moveToThread(this);
    if(taskType == THREAD_INIT_GRAPH){
        iftDataSet* dataset = mainwindow->workDataset;
        //mainwindow->tsne = iftCreateTsne(mainwindow->workDataset);
        //mainwindow->tsne->theta = 0.1;
        //iftComputeTsneProjection(mainwindow->tsne);
        if(dataset->projection == NULL){
            qDebug() << "computing projection";
            iftTsne* tsne = iftCreateTsne(dataset);
            iftComputeTsneProjection(tsne);
            iftDestroyTsne(&tsne);
        }
        float x,y;
        float rx = pointDefaultRx;
        float ry = pointDefaultRy;
        float scalinFactor_width = (mainwindow->myGraphicWidgetProjection->size().width()-(rx*4));
        float scalinFactor_height = (mainwindow->myGraphicWidgetProjection->size().height()-(ry*4));
        float factor;
        if(scalinFactor_height < scalinFactor_width){
            mainwindow->sceneScalingFactor = scalinFactor_height*2;
            factor = mainwindow->sceneScalingFactor;
            mainwindow->sceneScalingFactor = scalinFactor_height*2 + ry*4;
            //factor = scalinFactor_height*2;
        }else{
            //factor = scalinFactor_width*2;
            mainwindow->sceneScalingFactor = scalinFactor_width*2;
            factor = mainwindow->sceneScalingFactor;
            mainwindow->sceneScalingFactor = scalinFactor_width*2 + rx*4;
        }
        QString sampleName;
        unsigned long uniqueId;
        iftFileSet* fileSet = mainwindow->fileSet;
        if(fileSet == NULL){
            for (int i = 0; i < dataset->nsamples; ++i) {
                x = (dataset->projection->val[i*2])*factor  + rx*2;
                y = (dataset->projection->val[i*2 + 1])*factor + ry*2;
                MyNode *node = new MyNode();
                //sampleName = mainwindow->mountSampleName(&dataset2D->sample[i]);
                sampleName = QString::number(dataset->sample[i].truelabel) + "|"
                        + QString::number(dataset->sample[i].id);
                node->setPos(qreal(x),qreal(y));
                node->text  = QString("%1").arg(dataset->sample[i].truelabel);
                node->sampleName = sampleName;
                node->sampleImagePath = "";
                node->mountToopTip();
                node->setFlag(QGraphicsItem::ItemIsSelectable);
                //node->drawOption = optionText;
                mainwindow->graphNodes.push_back(node);
                //mainwindow->scene->addItem(node);
                uniqueId = dataset->sample[i].id;
                mainwindow->hashSampleId2GraphNodeRef.insert(uniqueId,node);
                mainwindow->hashGraphNodeRef2SampleId.insert(node,uniqueId);
                node->numberTimesChecked = dataset->sample[i].numberTimesChecked;
            }
        }else{
            QString imagePath;
            int id;
            QFileInfo *fileInfo = new QFileInfo();
            for (int i = 0; i < dataset->nsamples; ++i) {
                x = (dataset->projection->val[i*2])*factor  + rx*2;
                y = (dataset->projection->val[i*2 + 1])*factor + ry*2;
                id = dataset->sample[i].id;
                imagePath = fileSet->files[id]->path;
                MyNode *node = new MyNode();
                node->setPos(qreal(x),qreal(y));
                node->text  = QString("%1").arg(dataset->sample[i].truelabel);
                node->drawOption = optionText;

                QImage image(imagePath);
                if(image.isNull()){
                    node->sampleImagePath = ":/resource_images/Images/imageMissing.png";
                    fileInfo->setFile(imagePath);
                    node->sampleName = fileInfo->fileName();
                }else{
                    node->sampleImagePath = imagePath;
                    fileInfo->setFile(imagePath);
                    node->sampleName = fileInfo->fileName();
                }
                node->mountToopTip();
                node->setFlag(QGraphicsItem::ItemIsSelectable);
                mainwindow->graphNodes.push_back(node);
                //mainwindow->scene->addItem(node);
                uniqueId = dataset->sample[i].id;
                mainwindow->hashSampleId2GraphNodeRef.insert(uniqueId,node);
                mainwindow->hashGraphNodeRef2SampleId.insert(node,uniqueId);
                node->numberTimesChecked = dataset->sample[i].numberTimesChecked;
            }
            delete fileInfo;
        }
    }
    else if(taskType == THREAD_INIT_TREEVIEW){
        iftSelectorRDS *rds = (iftSelectorRDS*)(mainwindow->activeLearning->selector->selector);
        iftGenericVector* clusterOrganization = rds->clusterOfSamplesIndex;
        iftDataSet*  dataset = mainwindow->workDataset;
        QTreeWidget* treeWidget = mainwindow->treeWidgetSamples;
        int uniqueId;
        if(clusterOrganization != NULL){
            int sampleIndex;
            char name[20];
            for (size_t clusterIndex = 0; clusterIndex < clusterOrganization->size; ++clusterIndex) {
                QTreeWidgetItem *treeItem = new QTreeWidgetItem(treeWidget);
                treeItem->setText(0,tr("clusterId"));
                treeItem->setText(1,tr("name"));
                iftGenericVector* cluster = iftVectorAt(iftGenericVector*,
                                                        clusterOrganization,
                                                        clusterIndex);
                for (size_t i = 0; i < cluster->size; ++i) {
                    sampleIndex = iftVectorAt(int,cluster,i);
                    QTreeWidgetItem *treeItemChild = new QTreeWidgetItem(treeItem);
                    treeItemChild->setText(0,QString::number(clusterIndex));
                    sprintf(name,"%08d",dataset->sample[sampleIndex].id);
                    treeItemChild->setText(1,QString(name));
                    uniqueId = dataset->sample[sampleIndex].id;
                    mainwindow->hashSampleId2TreeItem.insert(uniqueId,treeItemChild);
                    mainwindow->hashTreeItem2SampleId.insert(treeItemChild,uniqueId);
                }
            }
        }
    }
    else if(taskType == THREAD_PROPAGATION_LABEL){
        if(mainwindow->workDataset == NULL){
            return;
        }
        if(mainwindow->hashSampleId2GraphNodeRef.isEmpty()){
            return;
        }

        for (int i = 0; i < mainwindow->workDataset->nsamples; ++i) {
            MyNode* node = mainwindow->hashSampleId2GraphNodeRef[mainwindow->workDataset->sample[i].id];
            if(node == NULL){
                continue;
            }
            if(node->isSelected()){
                mainwindow->workDataset->sample[i].truelabel = mainwindow->currentLabel;
                if(mainwindow->currentLabel == 0){
                    mainwindow->workDataset->sample[i].isSupervised = false;
                    mainwindow->workDataset->sample[i].status = IFT_TEST;
                }else{
                    mainwindow->workDataset->sample[i].isSupervised = true;
                    mainwindow->workDataset->sample[i].status = IFT_TRAIN;
                }
            }
        }

    }else if(taskType == THREAD_SEE_ONLY){
        if(mainwindow->workDataset == NULL){
            return;
        }
        if(mainwindow->hashSampleId2GraphNodeRef.isEmpty()){
            return;
        }
        if(mainwindow->filterCluster == 0 && mainwindow->filterClass == 0){
            for (int i = 0; i < mainwindow->workDataset->nsamples; ++i){
                MyNode* node = mainwindow->hashSampleId2GraphNodeRef[mainwindow->workDataset->sample[i].id];
                if(node == NULL){
                    continue;
                }
                node->setVisible(true);
            }
        }else if(mainwindow->filterCluster == 0 && mainwindow->filterClass != 0){
            for (int i = 0; i < mainwindow->workDataset->nsamples; ++i){
                MyNode* node = mainwindow->hashSampleId2GraphNodeRef[mainwindow->workDataset->sample[i].id];
                if(node == NULL){
                    continue;
                }
                if(mainwindow->workDataset->sample[i].label == mainwindow->filterClass){
                    node->setVisible(true);
                }else{
                    node->setVisible(false);
                }
            }
        }else if(mainwindow->filterCluster != 0 && mainwindow->filterClass == 0){
            for (int i = 0; i < mainwindow->workDataset->nsamples; ++i){
                MyNode* node = mainwindow->hashSampleId2GraphNodeRef[mainwindow->workDataset->sample[i].id];
                if(node == NULL){
                    continue;
                }
                if(mainwindow->workDataset->sample[i].group == mainwindow->filterCluster){
                    node->setVisible(true);
                }else{
                    node->setVisible(false);
                }
            }
        }else{
            for (int i = 0; i < mainwindow->workDataset->nsamples; ++i){
                MyNode* node = mainwindow->hashSampleId2GraphNodeRef[mainwindow->workDataset->sample[i].id];
                if(node == NULL){
                    continue;
                }
                if(mainwindow->workDataset->sample[i].label == mainwindow->filterClass){
                    if(mainwindow->workDataset->sample[i].group == mainwindow->filterCluster){
                        node->setVisible(true);
                    }else{
                        node->setVisible(false);
                    }
                }else{
                    node->setVisible(false);
                }
            }
        }
    }
    myFinished(taskType);
}




void MainWindow::on_pushButton_6_clicked()
{
    trainClassifier();
}

void MainWindow::trainClassifier(){
    if(workDataset == NULL){
        QMessageBox::critical(this,"Error","Dataset is empty");
        return;
    }
    if(activeLearning == NULL){
        QMessageBox::critical(this,"Error","Active learner is empty");
        return;
    }

    setTrainAndTestSamples();
    if(workDataset->ntrainsamples <= 0){
        return;
    }

    iftTrainGenericClassifier(activeLearning->classifier,workDataset);
    iftPredictGenericClassifier(activeLearning->classifier,workDataset);

    computeCrossValidationOnSupervisedSamples();
    if(developingMode == true){
        computeTrueLabelInfo();
    }
    updateTableWidget();
    numberOfSupervisedSamplesHistory[activeLearning->currentLearningCycle] = nSupervisedSamples;
    numberOfLabelPropagatedSamplesHistory[activeLearning->currentLearningCycle] = nPropagatedLabelSamples;
    learnCycleValues[activeLearning->currentLearningCycle] = activeLearning->currentLearningCycle;

    updateInfoArea();

    QString datasetName = QDir::currentPath() + QString("/learningCycle_datasets/")
            + QString("dataset") + QString::number(activeLearning->currentLearningCycle)
            + QString("_") + QString::number(operationNumber[activeLearning->currentLearningCycle])+
            QString(".zip");
    iftWriteDataSet(workDataset,datasetName.toLocal8Bit().constData());
    operationNumber[activeLearning->currentLearningCycle] += 1;
}

void MainWindow::on_actionSave_triggered()
{
    if(workDataset == NULL){
        QMessageBox::critical(this,"Error","Dataset is empty");
        return;
    }
    QString filename = QFileDialog::getSaveFileName(this,"Save scene"," ",".png");
    QImage image(scene->sceneRect().size().toSize(), QImage::Format_ARGB32);  // Create the image with the exact size of the shrunk scene
    image.fill(Qt::transparent);                                              // Start all pixels transparent
    QPainter painter(&image);
    scene->render(&painter);
    image.save(filename.toLocal8Bit().constData());
}

void MainWindow::on_pushButton_7_clicked()
{
    if(seeTrueLabelInfo == false){
        seeTrueLabelInfo = true;
    }else{
        seeTrueLabelInfo = false;
    }
    updateInfoArea();
    updateTableWidget();
}

void MainWindow::on_tableWidget_doubleClicked(const QModelIndex &index)
{
    Q_UNUSED(index)
    if(workDataset == NULL){
        return;
    }
    QTableWidget* table = new QTableWidget();
    table->clear();
    table->setRowCount(workDataset->nclasses);
    table->setColumnCount(workDataset->nclasses);

    QStringList headers;
    for (int i = 1; i < hashLabelId2LabelName.size(); ++i) {
        headers << hashLabelId2LabelName[i];
    }
    table->setHorizontalHeaderLabels(headers);
    table->setVerticalHeaderLabels(headers);

    for (int row = 0; row < workDataset->nclasses; ++row) {
        for (int col = 0; col < workDataset->nclasses; ++col) {
            QLabel *label = new QLabel("0.0000");
            QHBoxLayout* layout = new QHBoxLayout();
            layout->setAlignment( Qt::AlignCenter );
            layout->setSpacing(0);
            label->setLayout(layout);
            table->setCellWidget(row,col,label);
        }
    }

    float acc;
    QString text;
    iftMatrix* usedMatrix;
    if(seeTrueLabelInfo == true){
        usedMatrix = confusionMatrixNormalized;
    }else{
        usedMatrix = confusionMatrixSmallDatasetMean;
    }

    for (int row = 0; row < workDataset->nclasses; ++row) {
        for (int col = 0; col < workDataset->nclasses; ++col) {
            acc = iftMatrixElem(usedMatrix,col,row);
            float std = iftMatrixElem(confusionMatrixSmallDatasetStandardDeviation,col,row);
            QWidget * tableItem= table->cellWidget(row,col);
            QLabel* label = dynamic_cast<QLabel*>(tableItem);
            QFont font = label->font();
            font.setPointSize(8);
            label->setFont(font);
            text = QString::number(acc,10,4);
            if(seeTrueLabelInfo == false){
                text +=  QString(" +- ") + QString::number(std,10,4);
            }
            label->setText(text);
        }
    }
    //    table->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
    //    table->show();
    //    table->setAttribute( Qt::WA_DeleteOnClose );
    //    table->setWindowTitle("Confusion Matrix");

    QWidget *window = new QWidget;
    QHBoxLayout *layout = new QHBoxLayout;
    layout->addWidget(table);
    window->setLayout(layout);
    window->show();
    window->setAttribute(Qt::WA_DeleteOnClose);
    window->setWindowTitle("Confusion Matrix");
    window->setGeometry(0,0,100*workDataset->nclasses,100.0*workDataset->nclasses/2.0);

}



void MainWindow::on_pushButton_9_clicked()
{
    go2NextAciveLeaningState();
}

void MainWindow::on_pushButton_8_clicked()
{
    go2PreviousAciveLeaningState();
}

void MainWindow::on_actionScene_Options_triggered()
{
    if(windowSceneOptions->isHidden() == true){//NO
        windowSceneOptions->show();
    }else{
        windowSceneOptions->activateWindow();
    }
}

void MainWindow::windowSceneOptionsClosed(){

}
