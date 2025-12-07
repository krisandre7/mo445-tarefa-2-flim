#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "graphedge.h"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow)
{
    qDebug();

    ui->setupUi(this);
    initScene();
    initGraphicsView();
    enableWidgetsByMode("beforeProjection");
    updateNumbSamplesShown();
    currentNodeColorOption = ui->comboBoxNodeColor->currentText();
    drawShadowInPoints = false;

    //connect(windowSceneOptions,SIGNAL(sceneAutoProjectionChanged(bool)),ui->graphicsView,SLOT(updateSceneAutoReprojection(bool)));
    ui->projGraphicsView->autoReprojection = false;
    currentSupervisionField = ui->comboBoxSupervisionField->currentText();
    currentSupervisionMode = ui->comboBoxSupervisionMode->currentText();

    threadTimer = new QTimer(this);
    connect(threadTimer, SIGNAL(timeout()), this, SLOT(timer_slot()));

    /* read last used imgDir */
    QFile imgDirFilePtr("lastWorkingDir.txt");
    if (imgDirFilePtr.open(QIODevice::ReadOnly)) {
        QTextStream stream(&imgDirFilePtr);
        QString line = stream.readLine();

        if(!line.isEmpty()) {
            if(QDir(line).exists())
                workingDir = line;
            else
                workingDir = QDir::currentPath();
        } else {
            workingDir = QDir::currentPath();
        }
        imgDirFilePtr.close();
    } else {
        workingDir = QDir::currentPath();
    }
    ui->checkBoxShowClassGroupColorOnTooltip->hide();
}

MainWindow::~MainWindow()
{
    qDebug();

    QDir dir("thumbnails/");
    dir.removeRecursively();

    delete ui;
}

void MainWindow::initScene()
{
    qDebug();

    if(projGraphicsScene != NULL){
        projGraphicsScene->clear();
        delete projGraphicsScene;
    }

    projGraphicsScene = new QGraphicsScene(this);
    ui->projGraphicsView->setScene(projGraphicsScene);
}

void MainWindow::initGraphicsView()
{
    qDebug();

    ui->projGraphicsView->setDragMode(QGraphicsView::RubberBandDrag);
    ui->projGraphicsView->setInteractive(true);
    ui->projGraphicsView->setCacheMode(QGraphicsView::CacheBackground);
    ui->projGraphicsView->setViewportUpdateMode(QGraphicsView::SmartViewportUpdate);
}

void MainWindow::clearAllData(bool clearDatasets)
{
    qDebug();

    projGraphicsScene->clear();
    projGraphicsScene->update();
    graphNodes.clear();
    graphEdges.clear();
    KNNgraphEdges.clear();

    // clear hash tables
    hashGroupId2GroupColor.clear();
    hashGroupId2GroupName.clear();
    hashLabelId2LabelColor.clear();
    hashLabelId2LabelName.clear();
    hashGraphNodeRef2SampleId.clear();
    hashPath2SampleId.clear();

    // clear variables
    globalMinFeature = IFT_INFINITY_FLT;
    globalMaxFeature = IFT_INFINITY_FLT_NEG;
    globalMinFeatPerplexity = IFT_INFINITY_FLT;
    globalMaxFeatPerplexity = IFT_INFINITY_FLT_NEG;
    globalMinFeatMean = IFT_INFINITY_FLT;
    globalMaxFeatMean = IFT_INFINITY_FLT_NEG;
    globalMinFeatStdev= IFT_INFINITY_FLT;
    globalMaxFeatStdev = IFT_INFINITY_FLT_NEG;
    globalMinFeatWeight= IFT_INFINITY_FLT;
    globalMaxFeatWeight = IFT_INFINITY_FLT_NEG;
    //hashSampleId2CurrentItemInList.clear();
    //hashSampleId2CurrentSelected.clear();
    hashSampleId2GraphNodeRef.clear();

    // clear datasets
    if(clearDatasets) {
      if(originalDataSet != nullptr) iftDestroyDataSet(&originalDataSet);
      if(workingDataSet != nullptr)  iftDestroyDataSet(&workingDataSet);
      fileSet = nullptr;
    }
    updateNumbSamplesShown();

}

void MainWindow::createClassListView()
{
    qDebug();

    QStandardItemModel *model = new QStandardItemModel(workingDataSet->nclasses+1, 1);

    QStandardItem* item = new QStandardItem(QString("All"));
    item->setCheckable(true);
    item->setCheckState(Qt::Unchecked);
    model->setItem(0, item);

    for (int i = 1; i < workingDataSet->nclasses+1; i++)
    {
        QStandardItem* item = new QStandardItem(QString("%0").arg(i));

        item->setCheckable(true);
        item->setCheckState(Qt::Unchecked);
        item->setData(hashLabelId2LabelColor[i], Qt::DecorationRole);

        model->setItem(i, item);
    }
    ui->listViewSelectedClasses->setModel(model);
}

void MainWindow::createGroupListView()
{
    qDebug();

    QStandardItemModel *model = new QStandardItemModel(workingDataSet->ngroups+1, 1);

    QStandardItem* item = new QStandardItem(QString("All"));
    item->setCheckable(true);
    item->setCheckState(Qt::Unchecked);
    model->setItem(0, item);

    for (int i = 1; i < workingDataSet->ngroups+1; i++)
    {
        QStandardItem* item = new QStandardItem(QString("%0").arg(i));

        item->setCheckable(true);
        item->setCheckState(Qt::Unchecked);
        item->setData(hashGroupId2GroupColor[i], Qt::DecorationRole);

        model->setItem(i, item);
    }
    ui->listViewSelectedGroups->setModel(model);
}

void MainWindow::createSampleListView()
{
    qDebug();

    QStandardItemModel *model = new QStandardItemModel(workingDataSet->nsamples+1, 1);

    QStandardItem* item = new QStandardItem(QString("All"));
    item->setCheckable(true);
    item->setCheckState(Qt::Unchecked);
    item->setData(0);
    model->setItem(0, item);

    for (int i = 0; i < workingDataSet->nsamples; i++)
    {
        QStandardItem* item1 = new QStandardItem(QString("%1 (id: %2)").arg(i).arg(workingDataSet->sample[i].id));

        item1->setCheckable(true);
        item1->setCheckState(Qt::Unchecked);
        item1->setData(i);

        model->setItem(i+1, item1);
    }
    ui->listViewSelectedSamples->setModel(model);
}

void MainWindow::updateNumbSamplesShown()
{
    qDebug();

    numbSamplesShown = 0;
    numbSamplesSelected = 0;

    if(workingDataSet != NULL) {
        for (int i = 0; i < workingDataSet->nsamples; i++) {
            GraphNode* node = hashSampleId2GraphNodeRef[i];
            if(node == NULL)
                continue;
            if(node->isVisible())
                numbSamplesShown++;
            if(node->isSelected())
                numbSamplesSelected++;
        }
    }

    ui->lineEditNumbSamplesShown->setText(QString("Shown samples: %1").arg(numbSamplesShown) + QString(", currently selected samples (drag & drop): %1").arg(numbSamplesSelected));
}

void MainWindow::on_actionOpen_dataset_triggered()
{
    qDebug();

    dataSetName = QFileDialog::getOpenFileName(this, tr("Open dataset"), workingDir, tr("OPF dataset (*.zip);; All files (*.*)"));

    if (dataSetName.isEmpty())
        return;
    else {
        QFile file(dataSetName);
        if (!file.open(QIODevice::ReadOnly)) {
            QMessageBox::information(this, tr("Unable to open file"), file.errorString());
            return;
        }

        /* read dataset and perform t-SNE projection */
        clearAllData();
        originalDataSet = iftReadDataSet(dataSetName.toStdString().c_str());
        if(ui->checkBoxAutomaticPerplexity->isChecked())
            computeAutomaticPerplexity();
        updateDataSetInfoTextBoxes();
        tsneProjectionOnCurrentDataSet();
        
        fflush(stdout);
        createThumbnailOfCurrentDataSet();
        
        fflush(stdout);

        /* save the working directory */
        workingDir = QFileInfo(dataSetName).absolutePath();
        QFile workingDirFilePtr("lastWorkingDir.txt");
        if (workingDirFilePtr.open(QIODevice::WriteOnly)) {
            QTextStream stream(&workingDirFilePtr);
            stream << workingDir;
            workingDirFilePtr.close();
        }
    }
}


void MainWindow::updateDataSetInfoTextBoxes()
{
    qDebug();

    ui->lineEditDataSetName->setText(QString("Dataset: %1").
                                  arg(iftFilename(dataSetName.toStdString().c_str(), NULL)));

    iftDataSet *dataset = workingDataSet != NULL ? workingDataSet : originalDataSet;
    ui->lineEditDataSetInfo->setText(QString("Info: %1 samples, %2 feats, %3 classes, %4 groups, %5 train samples").
                                  arg(dataset->nsamples).
                                  arg(dataset->nfeats).
                                  arg(dataset->nclasses).
                                  arg(dataset->ngroups).
                                  arg(dataset->ntrainsamples));
    
}

void MainWindow::on_actionSave_dataset_triggered()
{
    qDebug();

    dataSetName = QFileDialog::getSaveFileName(this, tr("Save dataset"), workingDir, tr("OPF dataset (*.zip);; All files (*.*)"));

    if (!dataSetName.isEmpty()) {
        /* only the truelabel for the supervised samples is mantained, otherwise, the truelabel is 0 */
        for (int i = 0; i < workingDataSet->nsamples; ++i) {
            if(!iftHasSampleStatus(workingDataSet->sample[i], IFT_SUPERVISED) && !iftHasSampleStatus(workingDataSet->sample[i], IFT_LABELPROPAGATED)) {
                workingDataSet->sample[i].truelabel = 0;
                workingDataSet->sample[i].label = 0;
                workingDataSet->sample[i].group = 0;
            }
        }
        /* save the dataset */
        iftWriteDataSet(workingDataSet, dataSetName.toStdString().c_str());

        updateDataSetInfoTextBoxes();
        ui->projGraphicsView->paintNodes();
    }
}

void MainWindow::tsneProjectionOnCurrentDataSet()
{
    qDebug();

    if(originalDataSet == NULL)
        return;

    double perplexity = (double)ui->horizontalSliderPerplexity->value();
    if(originalDataSet->nsamples - 1 < 3 * perplexity) {
        QMessageBox::warning(this, tr("Error"), tr("Perplexity too large for the number of data points"), QMessageBox::Ok);
        return;
    }

    if(originalDataSet->ref_data_type == IFT_REF_DATA_FILESET)
        fileSet = (iftFileSet*)originalDataSet->ref_data;

    /* set timers */
    threadElapsedTime.start();
    threadTimer->start(100);

    /* create and run thread */
    enableWidgetsByMode("duringProjection");
    threadTaskType = THREAD_TSNE_PROJECTION;
    threadProjection = new MyThread(this, threadTaskType);
    threadProjection->start(QThread::HighestPriority);
    connect(threadProjection, SIGNAL(projectionFinished_signal(bool)), this, SLOT(projectionFinished_slot(bool)));
    connect(threadProjection, SIGNAL(finished()), threadProjection, SLOT(deleteLater()));
}

void MainWindow::createThumbnailOfCurrentDataSet()
{
    qDebug();

    if(originalDataSet == nullptr)
        return;

    if (fileSet == nullptr)
        return;
//        if(originalDataSet->ref_data_type == IFT_REF_DATA_FILESET)
//            fileSet = (iftFileSet*)originalDataSet->ref_data;

    printf("%ld\n",fileSet->n);
    fflush(stdout);
    thumbnailPath = iftCreateFileSet(fileSet->n);

    // set timers
    threadElapsedTime.start();
    threadTimer->start(100);

    threadTaskType = THREAD_THUMBNAIL;
    threadThumbnail = new MyThread(this, threadTaskType);
    threadThumbnail->start(QThread::HighestPriority);
    connect(threadThumbnail, SIGNAL(finished()), threadThumbnail, SLOT(deleteLater()));
}

void MainWindow::projectionFinished_slot(bool tsneExecuted)
{
    qDebug();
    
    ui->projGraphicsView->computeSceneScalingFactor();

    /* create hash tables and list views */
    createHashBetweenClassesAndColors();
    createHashBetweenGroupsAndColors();
    createClassListView();
    createGroupListView();
    createSampleListView();

    /* create nodes that will be added to the scene */
    qreal rx = pointDefaultRx;
    qreal ry = pointDefaultRy;
    qreal factor = sceneScalingFactor - rx*4;

    for (int i = 0; i < workingDataSet->nsamples; ++i) {
        /* create the graph node */
        GraphNode *node = new GraphNode(&workingDataSet->sample[i], workingDataSet->nfeats);

        /* sample position */
        qreal x = (workingDataSet->projection->val[i*2])*factor  + rx*2;
        qreal y = (workingDataSet->projection->val[i*2 + 1])*factor + ry*2;
        node->setPos(x, y);

        /* mount image tooltip (if fileset exists) */
        if(fileSet) {
            QString imagePath = thumbnailPath->files[workingDataSet->sample[i].id]->path;
            QImage image(imagePath);
            if (!image.isNull()) node->thumbnailImagePath = imagePath;
            else node->thumbnailImagePath = ":/resource_images/Images/imageMissing.png";
            imagePath = fileSet->files[workingDataSet->sample[i].id]->path;
            image = QImage(imagePath);            
            if (iftIsValidFormat(imagePath.toUtf8().data())) node->sampleImagePath = imagePath;
            else node->sampleImagePath = ":/resource_images/Images/imageMissing.png";
        }

        /* set other variables */
        node->setFlag(QGraphicsItem::ItemIsSelectable);
        node->numberTimesChecked = workingDataSet->sample[i].numberTimesChecked;

        /* initialize sample status */
        bool isSupervised = false, isLabelPropagated = false;
        if(iftHasSampleStatus(workingDataSet->sample[i], IFT_SUPERVISED)) isSupervised = true;
        if(iftHasSampleStatus(workingDataSet->sample[i], IFT_LABELPROPAGATED)) isLabelPropagated = true;
        iftSetSampleStatus(&workingDataSet->sample[i], IFT_TRAIN); // all the samples must have the IFT_TRAIN status
        if(isSupervised) iftAddSampleStatus(&workingDataSet->sample[i], IFT_SUPERVISED);
        if(isLabelPropagated) iftAddSampleStatus(&workingDataSet->sample[i], IFT_LABELPROPAGATED);

        /* insert the node in the vector of nodes (graph) and create corresponding hash tables */
        graphNodes.push_back(node);
        hashSampleId2GraphNodeRef.insert(i, node); // the sample's ID does not correspond to the dataset index. It must be used only as an index for the reference data
        hashGraphNodeRef2SampleId.insert(node, i); // the sample's ID does not correspond to the dataset index. It must be used only as an index for the reference data

        /* compute min and max values for node data */
        globalMinFeature = iftMin(globalMinFeature, iftMinFloatArray(node->samplePtr->feat, workingDataSet->nfeats));
        globalMaxFeature = iftMax(globalMaxFeature, iftMaxFloatArray(node->samplePtr->feat, workingDataSet->nfeats));
        globalMinFeatPerplexity = iftMin(globalMinFeatPerplexity, node->featPerplexity);
        globalMaxFeatPerplexity = iftMax(globalMaxFeatPerplexity, node->featPerplexity);
        globalMinFeatMean = iftMin(globalMinFeatMean, node->featMean);
        globalMaxFeatMean = iftMax(globalMaxFeatMean, node->featMean);
        globalMinFeatStdev = iftMin(globalMinFeatStdev, node->featStdev);
        globalMaxFeatStdev = iftMax(globalMaxFeatStdev, node->featStdev);
        globalMinFeatWeight = iftMin(globalMinFeatWeight, node->featWeight);
        globalMaxFeatWeight = iftMax(globalMaxFeatWeight, node->featWeight);
    }

    /* update info for each sample */
    for (int i = 0; i < workingDataSet->nsamples; ++i) {
        GraphNode* node = hashSampleId2GraphNodeRef[i];
        if(node == NULL){
            continue;
        }
        node->setMaxGlobalFeatPerplexity(globalMaxFeatPerplexity);
        node->setMinGlobalFeatPerplexity(globalMinFeatPerplexity);
        node->setMaxGlobalFeatMean(globalMaxFeatMean);
        node->setMinGlobalFeatMean(globalMinFeatMean);
        node->setMaxGlobalFeatStdev(globalMaxFeatStdev);
        node->setMinGlobalFeatStdev(globalMinFeatStdev);
        node->setMaxGlobalFeatWeight(globalMaxFeatWeight);
        node->setMinGlobalFeatWeight(globalMinFeatWeight);
        node->setDisplayNodeDataAsPointBorder(ui->checkBoxDisplayNodeDataFilter->isChecked());
        node->setCurrentNodeDataMode(ui->comboBoxNodeDataFilter->currentText());
    }

    /* add projected items to the scene */
    for (int i = 0; i < graphNodes.size(); ++i){
        //unsupervised
        if(!(iftHasSampleStatus(workingDataSet->sample[i], IFT_SUPERVISED))){
            graphNodes.at(i)->setFlag(QGraphicsItem::ItemIgnoresTransformations);
            projGraphicsScene->addItem(graphNodes.at(i));
            graphNodes.at(i)->Z = 0.0;
            graphNodes.at(i)->setZValue(0.0);
        }
    //for (int i = 0; i < graphNodes.size(); ++i)
        //supervised
        if(iftHasSampleStatus(workingDataSet->sample[i], IFT_SUPERVISED)){
            graphNodes.at(i)->setFlag(QGraphicsItem::ItemIgnoresTransformations);
            projGraphicsScene->addItem(graphNodes.at(i));
            graphNodes.at(i)->Z = 0.5;
            graphNodes.at(i)->setZValue(0.5);
        }
    }

    threadElapsedTime.start();
    threadTimer->start(100);

    /*
    threadTaskType = THREAD_GRAPH_EDGES;
    threadGraphEdges = new MyThread(this, threadTaskType);
    threadGraphEdges->start(QThread::HighPriority);
    connect(threadGraphEdges, SIGNAL(graphEdgesFinished_signal()), this, SLOT(graphEdgesFinished_slot()));
    connect(threadGraphEdges, SIGNAL(finished()), threadGraphEdges, SLOT(deleteLater()));
    */


    createGraphEdgesbykNN(); 
    //createHashBetweenIDAndGlobalkNNColor();

    /* update scene and paint nodes */
    projGraphicsScene->setSceneRect(0,0,sceneScalingFactor,sceneScalingFactor);
    projGraphicsScene->update();
    ui->projGraphicsView->paintNodes();

    /* update some ui elements */
    enableWidgetsByMode("afterProjection");
    createComboBoxForClassAndGroupFilters();
    setValuesForNodeDataFilterSpinBoxes();

    if(tsneExecuted)
        ui->statusBar->showMessage(QString("t-SNE projection created ... (Elapsed time: %1)").arg(formatTime(threadElapsedTime.elapsed())), 4000);
    else
        ui->statusBar->showMessage(QString("t-SNE projection loaded from the given dataset"), 4000);

    threadTimer->stop();

    updateDataSetInfoTextBoxes();

    updateNumbSamplesShown();

    updateSampleTooltip();

}

void MainWindow::graphEdgesFinished_slot()
{

}

void MainWindow::updateSampleTooltip()
{
    qDebug();

    for (int i = 0; i < workingDataSet->nsamples; ++i) {
        GraphNode* node = hashSampleId2GraphNodeRef[i];
        if(node == NULL){
            continue;
        }

        if(currentNodeColorOption == "True class")
            node->sampleName = QString("id: %1 | true class: %2").arg(workingDataSet->sample[i].id).arg(workingDataSet->sample[i].truelabel);
        else if(currentNodeColorOption == "Predicted")
            node->sampleName = QString("id: %1 | pred label: %2").arg(workingDataSet->sample[i].id).arg(workingDataSet->sample[i].label);
        else if(currentNodeColorOption == "Supervised")
            if(iftHasSampleStatus(workingDataSet->sample[i], IFT_SUPERVISED))
                node->sampleName = QString("id: %1 | sup class: %2").arg(workingDataSet->sample[i].id).arg(workingDataSet->sample[i].truelabel);
            else
                node->sampleName = QString("id: %1 | sup class: 0").arg(workingDataSet->sample[i].id);
        else if(currentNodeColorOption == "Propagated")
            if(iftHasSampleStatus(workingDataSet->sample[i], IFT_LABELPROPAGATED))
                node->sampleName = QString("id: %1 | prop class: %2").arg(workingDataSet->sample[i].id).arg(workingDataSet->sample[i].truelabel);
            else
                node->sampleName = QString("id: %1 | prop class: 0").arg(workingDataSet->sample[i].id);
        else if(currentNodeColorOption == "Supervised + propagated")
            if(iftHasSampleStatus(workingDataSet->sample[i], IFT_SUPERVISED) || iftHasSampleStatus(workingDataSet->sample[i], IFT_LABELPROPAGATED))
                node->sampleName = QString("id: %1 | sup+prop class: %2").arg(workingDataSet->sample[i].id).arg(workingDataSet->sample[i].truelabel);
            else
                node->sampleName = QString("id: %1 | sup+prop class: 0").arg(workingDataSet->sample[i].id);
        else if(currentNodeColorOption == "Group")
            node->sampleName = QString("id: %1 | group: %2").arg(workingDataSet->sample[i].id).arg(workingDataSet->sample[i].group);

        node->drawOption = nodeTypeToDrawOption(ui->comboBoxNodeType->currentText());
        node->mountToolTip();
    }
}

void MainWindow::setValuesForNodeDataFilterSpinBoxes()
{
    qDebug();

    if(ui->comboBoxNodeDataFilter->currentText() == "Perplexity") {
        ui->doubleSpinBoxNodeDataFilterFrom->setMinimum(globalMinFeatPerplexity - ui->doubleSpinBoxNodeDataFilterFrom->singleStep());
        ui->doubleSpinBoxNodeDataFilterFrom->setMaximum(globalMaxFeatPerplexity + ui->doubleSpinBoxNodeDataFilterFrom->singleStep());
        ui->doubleSpinBoxNodeDataFilterFrom->setValue(globalMinFeatPerplexity - ui->doubleSpinBoxNodeDataFilterFrom->singleStep());
        ui->doubleSpinBoxNodeDataFilterTo->setMinimum(globalMinFeatPerplexity - ui->doubleSpinBoxNodeDataFilterTo->singleStep());
        ui->doubleSpinBoxNodeDataFilterTo->setMaximum(globalMaxFeatPerplexity + ui->doubleSpinBoxNodeDataFilterTo->singleStep());
        ui->doubleSpinBoxNodeDataFilterTo->setValue(globalMaxFeatPerplexity + ui->doubleSpinBoxNodeDataFilterTo->singleStep());
    }
    else if(ui->comboBoxNodeDataFilter->currentText() == "Mean") {
        ui->doubleSpinBoxNodeDataFilterFrom->setMinimum(globalMinFeatMean - ui->doubleSpinBoxNodeDataFilterFrom->singleStep());
        ui->doubleSpinBoxNodeDataFilterFrom->setMaximum(globalMaxFeatMean + ui->doubleSpinBoxNodeDataFilterFrom->singleStep());
        ui->doubleSpinBoxNodeDataFilterFrom->setValue(globalMinFeatMean - ui->doubleSpinBoxNodeDataFilterFrom->singleStep());
        ui->doubleSpinBoxNodeDataFilterTo->setMinimum(globalMinFeatMean - ui->doubleSpinBoxNodeDataFilterTo->singleStep());
        ui->doubleSpinBoxNodeDataFilterTo->setMaximum(globalMaxFeatMean + ui->doubleSpinBoxNodeDataFilterTo->singleStep());
        ui->doubleSpinBoxNodeDataFilterTo->setValue(globalMaxFeatMean + ui->doubleSpinBoxNodeDataFilterTo->singleStep());
    }
    else if(ui->comboBoxNodeDataFilter->currentText() == "Stdev") {
        ui->doubleSpinBoxNodeDataFilterFrom->setMinimum(globalMinFeatStdev - ui->doubleSpinBoxNodeDataFilterFrom->singleStep());
        ui->doubleSpinBoxNodeDataFilterFrom->setMaximum(globalMaxFeatStdev + ui->doubleSpinBoxNodeDataFilterFrom->singleStep());
        ui->doubleSpinBoxNodeDataFilterFrom->setValue(globalMinFeatStdev - ui->doubleSpinBoxNodeDataFilterFrom->singleStep());
        ui->doubleSpinBoxNodeDataFilterTo->setMinimum(globalMinFeatStdev - ui->doubleSpinBoxNodeDataFilterTo->singleStep());
        ui->doubleSpinBoxNodeDataFilterTo->setMaximum(globalMaxFeatStdev + ui->doubleSpinBoxNodeDataFilterTo->singleStep());
        ui->doubleSpinBoxNodeDataFilterTo->setValue(globalMaxFeatStdev + ui->doubleSpinBoxNodeDataFilterTo->singleStep());
    }
    else if(ui->comboBoxNodeDataFilter->currentText() == "Weight") {
        ui->doubleSpinBoxNodeDataFilterFrom->setMinimum(globalMinFeatWeight - ui->doubleSpinBoxNodeDataFilterFrom->singleStep());
        ui->doubleSpinBoxNodeDataFilterFrom->setMaximum(globalMaxFeatWeight + ui->doubleSpinBoxNodeDataFilterFrom->singleStep());
        ui->doubleSpinBoxNodeDataFilterFrom->setValue(globalMinFeatWeight - ui->doubleSpinBoxNodeDataFilterFrom->singleStep());
        ui->doubleSpinBoxNodeDataFilterTo->setMinimum(globalMinFeatWeight - ui->doubleSpinBoxNodeDataFilterTo->singleStep());
        ui->doubleSpinBoxNodeDataFilterTo->setMaximum(globalMaxFeatWeight + ui->doubleSpinBoxNodeDataFilterTo->singleStep());
        ui->doubleSpinBoxNodeDataFilterTo->setValue(globalMaxFeatWeight + ui->doubleSpinBoxNodeDataFilterTo->singleStep());
    }

    ui->doubleSpinBoxFeaturesFrom->setMinimum(globalMinFeature - ui->doubleSpinBoxFeaturesFrom->singleStep());
    ui->doubleSpinBoxFeaturesFrom->setMaximum(globalMaxFeature + ui->doubleSpinBoxFeaturesFrom->singleStep());
    ui->doubleSpinBoxFeaturesFrom->setValue(globalMinFeature - ui->doubleSpinBoxFeaturesFrom->singleStep());
    ui->doubleSpinBoxFeaturesTo->setMinimum(globalMinFeature - ui->doubleSpinBoxFeaturesTo->singleStep());
    ui->doubleSpinBoxFeaturesTo->setMaximum(globalMaxFeature + ui->doubleSpinBoxFeaturesTo->singleStep());
    ui->doubleSpinBoxFeaturesTo->setValue(globalMaxFeature + ui->doubleSpinBoxFeaturesTo->singleStep());
}


void MainWindow::enableWidgetsByMode(QString mode)
{
    qDebug();

    if(mode == "beforeProjection") {
        ui->projGraphicsView->setEnabled(false);
        ui->checkBoxAutomaticPerplexity->setEnabled(true);
        ui->horizontalSliderNumIterTSNE->setEnabled(true);
        ui->spinBoxNumIterTSNE->setEnabled(true);
        ui->pushButtonProjectTSNE->setEnabled(true);
        ui->pushButtonInvertSelection->setEnabled(false);
        ui->comboBoxNodeType->setEnabled(false);
        ui->comboBoxNodeColor->setEnabled(false);
        ui->comboBoxFilterClass->setEnabled(false);
        ui->comboBoxFilterGroup->setEnabled(false);
        //ui->comboBoxSupervisionMode->setEnabled(false);
        ui->comboBoxSupervisionField->setEnabled(false);
        ui->pushButtonResetSupervision->setEnabled(false);
        //ui->comboBoxClassClustAction->setEnabled(false);
        //ui->comboBoxClassClustMethod->setEnabled(false);
        //ui->pushButtonClassClustExecute->setEnabled(false);
        //ui->pushButtonClassClustDisplay->setEnabled(false);
        ui->pushButtonSaveCanvas->setEnabled(false);
        ui->pushButtonChangeBkgColor->setEnabled(false);
        ui->pushButtonEnableShadow->setEnabled(false);
        ui->comboBoxNodeDataFilter->setEnabled(false);
        ui->doubleSpinBoxNodeDataFilterFrom->setEnabled(false);
        ui->doubleSpinBoxNodeDataFilterTo->setEnabled(false);
        ui->doubleSpinBoxFeaturesFrom->setEnabled(false);
        ui->doubleSpinBoxFeaturesTo->setEnabled(false);
        ui->checkBoxDisplayNodeDataFilter->setEnabled(false);
        ui->checkBoxInverseClassFilter->setEnabled(false);
        ui->checkBoxInverseGroupFilter->setEnabled(false);
        ui->listViewSelectedClasses->setEnabled(false);
        ui->listViewSelectedGroups->setEnabled(false);
        ui->listViewSelectedSamples->setEnabled(false);
        ui->pushButtonSaveSelectedClasses->setEnabled(false);
        ui->pushButtonOpenSelectedClasses->setEnabled(false);
        ui->pushButtonSaveSelectedGroups->setEnabled(false);
        ui->pushButtonOpenSelectedGroups->setEnabled(false);
        ui->checkBoxShowClassGroupColorOnTooltip->setEnabled(false);
    }

    if(mode == "duringProjection") {
        ui->projGraphicsView->setEnabled(false);
        ui->checkBoxAutomaticPerplexity->setEnabled(false);
        ui->horizontalSliderNumIterTSNE->setEnabled(false);
        ui->spinBoxNumIterTSNE->setEnabled(false);
        ui->pushButtonProjectTSNE->setEnabled(false);
        ui->pushButtonInvertSelection->setEnabled(false);
        ui->comboBoxNodeType->setEnabled(false);
        ui->comboBoxNodeColor->setEnabled(false);
        ui->comboBoxFilterClass->setEnabled(false);
        ui->comboBoxFilterGroup->setEnabled(false);
        //ui->comboBoxSupervisionMode->setEnabled(false);
        ui->comboBoxSupervisionField->setEnabled(false);
        ui->pushButtonResetSupervision->setEnabled(false);
        //ui->comboBoxClassClustAction->setEnabled(false);
        //ui->comboBoxClassClustMethod->setEnabled(false);
        //ui->pushButtonClassClustExecute->setEnabled(false);
        //ui->pushButtonClassClustDisplay->setEnabled(false);
        ui->pushButtonSaveCanvas->setEnabled(false);
        ui->pushButtonChangeBkgColor->setEnabled(false);
        ui->pushButtonEnableShadow->setEnabled(false);
        ui->comboBoxNodeDataFilter->setEnabled(false);
        ui->doubleSpinBoxNodeDataFilterFrom->setEnabled(false);
        ui->doubleSpinBoxNodeDataFilterTo->setEnabled(false);
        ui->doubleSpinBoxFeaturesFrom->setEnabled(false);
        ui->doubleSpinBoxFeaturesTo->setEnabled(false);
        ui->checkBoxDisplayNodeDataFilter->setEnabled(false);
        ui->checkBoxInverseClassFilter->setEnabled(false);
        ui->checkBoxInverseGroupFilter->setEnabled(false);
        ui->listViewSelectedClasses->setEnabled(false);
        ui->listViewSelectedGroups->setEnabled(false);
        ui->listViewSelectedSamples->setEnabled(false);
        ui->pushButtonSaveSelectedClasses->setEnabled(false);
        ui->pushButtonOpenSelectedClasses->setEnabled(false);
        ui->pushButtonSaveSelectedGroups->setEnabled(false);
        ui->pushButtonOpenSelectedGroups->setEnabled(false);
        ui->checkBoxShowClassGroupColorOnTooltip->setEnabled(false);
    }

    if(mode == "afterProjection") {
        ui->projGraphicsView->setEnabled(true);
        ui->checkBoxAutomaticPerplexity->setEnabled(true);
        ui->horizontalSliderNumIterTSNE->setEnabled(true);
        ui->spinBoxNumIterTSNE->setEnabled(true);
        ui->pushButtonProjectTSNE->setEnabled(true);
        ui->pushButtonInvertSelection->setEnabled(true);
        ui->comboBoxNodeType->setEnabled(true);
        ui->comboBoxNodeColor->setEnabled(true);
        ui->comboBoxFilterClass->setEnabled(true);
        ui->comboBoxFilterGroup->setEnabled(true);
        //ui->comboBoxSupervisionMode->setEnabled(true);
        ui->comboBoxSupervisionField->setEnabled(true);
        ui->pushButtonResetSupervision->setEnabled(true);
        //ui->comboBoxClassClustAction->setEnabled(false);
        //ui->comboBoxClassClustMethod->setEnabled(false);
        //ui->pushButtonClassClustExecute->setEnabled(false);
        //ui->pushButtonClassClustDisplay->setEnabled(false);
        ui->pushButtonSaveCanvas->setEnabled(true);
        ui->pushButtonChangeBkgColor->setEnabled(true);
        ui->pushButtonEnableShadow->setEnabled(true);
        ui->comboBoxNodeDataFilter->setEnabled(true);
        ui->doubleSpinBoxNodeDataFilterFrom->setEnabled(true);
        ui->doubleSpinBoxNodeDataFilterTo->setEnabled(true);
        ui->doubleSpinBoxFeaturesFrom->setEnabled(true);
        ui->doubleSpinBoxFeaturesTo->setEnabled(true);
        ui->checkBoxDisplayNodeDataFilter->setEnabled(true);
        ui->checkBoxInverseClassFilter->setEnabled(true);
        ui->checkBoxInverseGroupFilter->setEnabled(true);
        ui->listViewSelectedClasses->setEnabled(true);
        ui->listViewSelectedGroups->setEnabled(true);
        ui->listViewSelectedSamples->setEnabled(true);
        ui->pushButtonSaveSelectedClasses->setEnabled(true);
        ui->pushButtonOpenSelectedClasses->setEnabled(true);
        ui->pushButtonSaveSelectedGroups->setEnabled(true);
        ui->pushButtonOpenSelectedGroups->setEnabled(true);
        ui->checkBoxShowClassGroupColorOnTooltip->setEnabled(true);
    }
}

void MainWindow::createComboBoxForClassAndGroupFilters()
{
    qDebug();

    /* class filter */
    ui->comboBoxFilterClass->clear();
    for (int j = 0; j < hashLabelId2LabelName.size(); j++) {
        ui->comboBoxFilterClass->addItem(hashLabelId2LabelName[j], hashLabelId2LabelColor[j]);
        if(j > 0) {
            const QModelIndex idx = ui->comboBoxFilterClass->model()->index(j, 0);
            ui->comboBoxFilterClass->model()->setData(idx, hashLabelId2LabelColor[j], Qt::DecorationRole);
        }
    }
    if(hashLabelId2LabelName.size() > 0)
        ui->comboBoxFilterClass->setItemText(0, tr("All"));
    else
        ui->comboBoxFilterClass->addItem("All", QColor(0, 0, 0));
    ui->comboBoxFilterClass->setCurrentIndex(0);

    /* group filter */
    ui->comboBoxFilterGroup->clear();
    for (int j = 0; j < hashGroupId2GroupName.size(); j++) {
        ui->comboBoxFilterGroup->addItem(hashGroupId2GroupName[j], hashGroupId2GroupColor[j]);
        if(j > 0) {
            const QModelIndex idx = ui->comboBoxFilterGroup->model()->index(j, 0);
            ui->comboBoxFilterGroup->model()->setData(idx, hashGroupId2GroupColor[j], Qt::DecorationRole);
        }
    }
    if(hashGroupId2GroupName.size() > 0)
        ui->comboBoxFilterGroup->setItemText(0, tr("All"));
    else
        ui->comboBoxFilterGroup->addItem("All", QColor(0, 0, 0));
    ui->comboBoxFilterGroup->setCurrentIndex(0);
}

void MainWindow::createHashBetweenClassesAndColors()
{
    qDebug();

    if(workingDataSet == NULL)
        return;

    hashLabelId2LabelColor.clear();
    hashLabelId2LabelName.clear();

    if(workingDataSet->nclasses <= 0) {
        if(QMessageBox::warning(this, tr("Warning"), tr("The dataset does not contain the number of classes set. Do you want to count the classes present?"),
                                QMessageBox::Yes | QMessageBox::No) == QMessageBox::Yes)
            iftCountNumberOfClassesDataSet(workingDataSet);
    }
    int nclasses = workingDataSet->nclasses;

    if(nclasses > 0) {
        hashLabelId2LabelName.insert(0, "Unknown");
        hashLabelId2LabelColor.insert(0, QColor(0,0,0,200));

        /* create a color table */
        iftColorTable *ctb = iftCategoricalColorTable(nclasses);

        /* create new colors according to the number of classes */
        for (int i = 0; i < nclasses; i++) {
            iftColor RGB = iftYCbCrtoRGB(ctb->color[i], 255);
            QColor color;
            color.setRgb(RGB.val[0], RGB.val[1], RGB.val[2]);
            color.setAlpha(200);
            hashLabelId2LabelColor.insert(i+1, QColor(color));
            QString trueLabelName = tr("Class ") + QString::number(i+1);
            hashLabelId2LabelName.insert(i+1, trueLabelName);
        }
        iftDestroyColorTable(&ctb);
    }
}

void MainWindow::createHashBetweenGroupsAndColors()
{
    qDebug();

    if(workingDataSet == NULL)
        return;

    hashGroupId2GroupColor.clear();
    hashGroupId2GroupName.clear();

    if(workingDataSet->ngroups <= 0) {
        if(QMessageBox::warning(this, tr("Warning"), tr("The dataset does not contain the number of groups set. Do you want to count the groups present?"),
                                QMessageBox::Yes | QMessageBox::No) == QMessageBox::Yes)
            iftCountNumberOfGroupsDataSet(workingDataSet);
    }
    int ngroups = workingDataSet->ngroups;

    if(ngroups> 0) {
        hashGroupId2GroupName.insert(0, "Unknown");
        hashGroupId2GroupColor.insert(0, QColor(0,0,0,200));

        /* create a color table */
        iftColorTable *ctb = iftCategoricalColorTable(ngroups);

        /* create new colors according to the number of groupss */
        for (int i = 0; i < ngroups; i++) {
            iftColor RGB = iftYCbCrtoRGB(ctb->color[i], 255);
            QColor color;
            color.setRgb(RGB.val[0], RGB.val[1], RGB.val[2]);
            color.setAlpha(200);
            hashGroupId2GroupColor.insert(i+1, QColor(color));
            QString groupName = tr("Group ") + QString::number(i+1);
            hashGroupId2GroupName.insert(i+1, groupName);
        }
        iftDestroyColorTable(&ctb);
    }
}

void MainWindow::addNewClassForSupervision()
{
    qDebug();

    workingDataSet->nclasses++;
    createHashBetweenClassesAndColors();
    createComboBoxForClassAndGroupFilters();
    projGraphicsScene->update();
    updateDataSetInfoTextBoxes();
}

void MainWindow::addNewGroupForSupervision()
{
    qDebug();

    workingDataSet->ngroups++;
    createHashBetweenGroupsAndColors();
    createComboBoxForClassAndGroupFilters();
    projGraphicsScene->update();
    updateDataSetInfoTextBoxes();
}

void MainWindow::addClassToSelectedClassesListView()
{
    qDebug();

    QStandardItemModel *selClassesModel = (QStandardItemModel*)ui->listViewSelectedClasses->model();
    QStandardItemModel *selSamplesModel = (QStandardItemModel*)ui->listViewSelectedSamples->model();

    /* set the selected classes as checked in the list view */
    for(int i = 0; i < workingDataSet->nsamples; i++) {
        GraphNode* node = hashSampleId2GraphNodeRef[i];
        if(node == NULL){
            continue;
        }
        if(node->isSelected()) {
            int classId = workingDataSet->sample[i].truelabel;
            if(classId != 0) {
                QStandardItemModel *model = (QStandardItemModel*)ui->listViewSelectedClasses->model();
                model->item(classId)->setCheckState(Qt::Checked);
            }
        }
    }

    /* set the samples of the selected classes as checked */
    for(int i = 0; i < workingDataSet->nsamples; i++) {
        int label = workingDataSet->sample[i].truelabel;
        selSamplesModel->item(i+1)->setCheckState(selClassesModel->item(label)->checkState());
    }

    /* set all the samples (graphical items) as unselected (to avoid conflicts with the funtion that verifies the checkboxes) */
    for(int i = 0; i < workingDataSet->nsamples; i++) {
        GraphNode* node = hashSampleId2GraphNodeRef[i];
        if(node == NULL){
            continue;
        }
        node->setSelected(false);
    }

    markSamplesAsSelectedByClassOrGroup();
    updateLabelsSelectedClassesGroupsSamples();
    verifyCheckBoxesInListViews();
    updateNumbSamplesShown();
}


void MainWindow::SelectGroupUp()
{
qDebug();
int filterGroup = ui->comboBoxFilterGroup->currentIndex();
    if (filterGroup < (ui->comboBoxFilterGroup->count()-1))
        ui->comboBoxFilterGroup->setCurrentIndex(filterGroup+1);

}

void MainWindow::SelectGroupDown()
{
qDebug();
int filterGroup = ui->comboBoxFilterGroup->currentIndex();
if (filterGroup>0)
    ui->comboBoxFilterGroup->setCurrentIndex(filterGroup-1);

}

void MainWindow::addGroupToSelectedGroupsListView()
{
    qDebug();

    QStandardItemModel *selGroupsModel = (QStandardItemModel*)ui->listViewSelectedGroups->model();
    QStandardItemModel *selSamplesModel = (QStandardItemModel*)ui->listViewSelectedSamples->model();
    int filterGroup = ui->comboBoxFilterGroup->currentIndex();
    QStandardItemModel *model = (QStandardItemModel*)ui->listViewSelectedGroups->model();

    if (model->item(filterGroup)->checkState()==Qt::Checked)
       model->item(filterGroup)->setCheckState(Qt::Unchecked);
    else
        model->item(filterGroup)->setCheckState(Qt::Checked);



    /* set the selected groups as checked in the list view */
    for(int i = 0; i < workingDataSet->nsamples; i++) {
        GraphNode* node = hashSampleId2GraphNodeRef[i];
        if(node == NULL){
            continue;
        }
        if(node->isSelected()) {
            int groupId = workingDataSet->sample[i].group;
            if(groupId != 0) {
                model = (QStandardItemModel*)ui->listViewSelectedGroups->model();
                model->item(groupId)->setCheckState(Qt::Checked);
            }
        }
    }

    /* set the samples of the selected groups as checked */
    for(int i = 0; i < workingDataSet->nsamples; i++) {
        int group = workingDataSet->sample[i].group;
        selSamplesModel->item(i+1)->setCheckState(selGroupsModel->item(group)->checkState());
    }

    /* set all the samples (graphical items) as unselected (to avoid conflicts with the funtion that verifies the checkboxes) */
    for(int i = 0; i < workingDataSet->nsamples; i++) {
        GraphNode* node = hashSampleId2GraphNodeRef[i];
        if(node == NULL){
            continue;
        }
        node->setSelected(false);
    }

    markSamplesAsSelectedByClassOrGroup();
    updateLabelsSelectedClassesGroupsSamples();
    verifyCheckBoxesInListViews();
    updateNumbSamplesShown();
}

void MainWindow::addSampleToSelectedSamplesListView()
{
    qDebug();

    /* set the selected samples as checked in the list view */
    for(int i = 0; i < workingDataSet->nsamples; i++) {
        GraphNode* node = hashSampleId2GraphNodeRef[i];
        if(node == NULL){
            continue;
        }
        if(node->isSelected()) {
            QStandardItemModel *model = (QStandardItemModel*)ui->listViewSelectedSamples->model();
            model->item(i+1)->setCheckState(Qt::Checked);
        }
    }

    /* set all the samples (graphical items) as unselected (to avoid conflicts with the funtion that verifies the checkboxes) */
    for(int i = 0; i < workingDataSet->nsamples; i++) {
        GraphNode* node = hashSampleId2GraphNodeRef[i];
        if(node == NULL){
            continue;
        }
        node->setSelected(false);
    }

    markSamplesAsSelectedByClassOrGroup();
    updateLabelsSelectedClassesGroupsSamples();
    verifyCheckBoxesInListViews();
    updateNumbSamplesShown();
}

void MainWindow::markSamplesAsSelectedByClassOrGroup()
{
    qDebug();

    QStandardItemModel *selClassesModel = (QStandardItemModel*)ui->listViewSelectedClasses->model();
    QStandardItemModel *selGroupsModel = (QStandardItemModel*)ui->listViewSelectedGroups->model();
    QStandardItemModel *selSamplesModel = (QStandardItemModel*)ui->listViewSelectedSamples->model();

    int *selSamplPerClass = iftAllocIntArray(workingDataSet->nclasses+1);
    int *selSamplPerGroup = iftAllocIntArray(workingDataSet->ngroups+1);

    for(int i = 0; i < workingDataSet->nsamples; i++) {
        GraphNode* node = hashSampleId2GraphNodeRef[i];
        if(node == NULL){
            continue;
        }
        if(currentNodeColorOption == "True class") {
            int selClass = workingDataSet->sample[i].truelabel;
            if(selClassesModel->item(selClass)->checkState() == Qt::Checked)
                node->isSelectedByClassOrGroup = true;
            else
                node->isSelectedByClassOrGroup = false;
        }
        else if(currentNodeColorOption == "Group") {
            int selGroup = workingDataSet->sample[i].group;
            if(selGroupsModel->item(selGroup)->checkState() == Qt::Checked)
                node->isSelectedByClassOrGroup = true;
            else
                node->isSelectedByClassOrGroup = false;
        }

        if(selSamplesModel->item(i+1)->checkState() == Qt::Checked) {
            selSamplPerClass[workingDataSet->sample[i].truelabel]++;
            selSamplPerGroup[workingDataSet->sample[i].group]++;
            node->isSelectedByClassOrGroup = true;
        }
        else
            node->isSelectedByClassOrGroup = false;
    }

    /* verify if all the samples in a class or group were selected */
    int *sampPerClass = iftCountSamplesPerClassDataSet(workingDataSet);
    int *sampPerGroup = iftCountSamplesPerGroupDataSet(workingDataSet);

    for(int i = 1; i < workingDataSet->nclasses+1; i++) {
        if(sampPerClass[i] != 0 && selSamplPerClass[i] == sampPerClass[i])
            selClassesModel->item(i)->setCheckState(Qt::Checked);
        else
            selClassesModel->item(i)->setCheckState(Qt::Unchecked);
    }

    for(int i = 1; i < workingDataSet->ngroups+1; i++) {
        if(sampPerGroup[i] != 0 && selSamplPerGroup[i] == sampPerGroup[i])
            selGroupsModel->item(i)->setCheckState(Qt::Checked);
        else
            selGroupsModel->item(i)->setCheckState(Qt::Unchecked);
    }

    ui->projGraphicsView->paintNodes();
    projGraphicsScene->update();
}

void MainWindow::updateLabelsSelectedClassesGroupsSamples()
{
    qDebug();

    QStandardItemModel *selClassesModel = (QStandardItemModel*)ui->listViewSelectedClasses->model();
    QStandardItemModel *selGroupsModel = (QStandardItemModel*)ui->listViewSelectedGroups->model();
    QStandardItemModel *selSamplesModel = (QStandardItemModel*)ui->listViewSelectedSamples->model();

    int nClasses = 0;
    for(int i = 1; i < workingDataSet->nclasses+1; i++)
        if(selClassesModel->item(i)->checkState() == Qt::Checked) {
            nClasses++;
        }

    int nGroups = 0;
    for(int i = 1; i < workingDataSet->ngroups+1; i++)
        if(selGroupsModel->item(i)->checkState() == Qt::Checked) {
            nGroups++;
        }

    int nSamples = 0;
    for(int i = 1; i < workingDataSet->nsamples+1; i++) {
        if(selSamplesModel->item(i)->checkState() == Qt::Checked) {
            nSamples++;
        }
    }

    ui->labelSelectedClasses->setText(tr("Sel. classes: %1").arg(nClasses));
    ui->labelSelectedGroups->setText(tr("Sel. groups: %1").arg(nGroups));
    ui->labelSelectedSamples->setText(tr("Sel. samples: %1").arg(nSamples));
}

void MainWindow::verifyCheckBoxesInListViews()
{
    qDebug();

    QStandardItemModel *selClassesModel = (QStandardItemModel*)ui->listViewSelectedClasses->model();
    QStandardItemModel *selGroupsModel = (QStandardItemModel*)ui->listViewSelectedGroups->model();
    QStandardItemModel *selSamplesModel = (QStandardItemModel*)ui->listViewSelectedSamples->model();

    /* verify the selected classes */
    int nSelClasses = 0;
    for(int i = 1; i < workingDataSet->nclasses+1; i++)
        if(selClassesModel->item(i)->checkState() == Qt::Checked)
            nSelClasses++;
    if(nSelClasses == workingDataSet->nclasses)
        selClassesModel->item(0)->setCheckState(Qt::Checked);
    else
        selClassesModel->item(0)->setCheckState(Qt::Unchecked);

    /* verify the selected groups */
    int nSelGroups = 0;
    for(int i = 1; i < workingDataSet->ngroups+1; i++)
        if(selGroupsModel->item(i)->checkState() == Qt::Checked)
            nSelGroups++;
    if(nSelGroups == workingDataSet->ngroups)
        selGroupsModel->item(0)->setCheckState(Qt::Checked);
    else
        selGroupsModel->item(0)->setCheckState(Qt::Unchecked);

    /* verify the selected samples */
    int nSelSamples = 0;
    for(int i = 1; i < workingDataSet->nsamples+1; i++)
        if(selSamplesModel->item(i)->checkState() == Qt::Checked)
            nSelSamples++;
    if(nSelSamples == workingDataSet->nsamples)
        selSamplesModel->item(0)->setCheckState(Qt::Checked);
    else
        selSamplesModel->item(0)->setCheckState(Qt::Unchecked);
}

void MainWindow::on_horizontalSliderPerplexity_valueChanged(int value)
{
    qDebug();

    ui->spinBoxPerplexity->setValue(value);
}

void MainWindow::on_horizontalSliderNumIterTSNE_valueChanged(int value)
{
    qDebug();

    ui->spinBoxNumIterTSNE->setValue(value);
}

void MainWindow::on_spinBoxPerplexity_valueChanged(int arg1)
{
    qDebug();

    ui->horizontalSliderPerplexity->setValue(arg1);
}

void MainWindow::on_spinBoxNumIterTSNE_valueChanged(int arg1)
{
    qDebug();

    ui->horizontalSliderNumIterTSNE->setValue(arg1);
}

void MainWindow::on_pushButtonProjectTSNE_clicked()
{
    qDebug();

    if(originalDataSet == NULL)
        if(QMessageBox::warning(this, tr("Warning"), tr("A dataset must be loaded to perform t-SNE"), QMessageBox::Ok) == QMessageBox::Ok)
            return;

    /* clear data and perform t-SNE projection again */
    clearAllData(false);

    if(originalDataSet->projection != NULL)
        iftDestroyDoubleMatrix(&originalDataSet->projection);

    tsneProjectionOnCurrentDataSet();
}



void MainWindow::on_pushButtonInvertSelection_clicked()
{
    qDebug();
    QStandardItemModel *selSamplesModel = (QStandardItemModel*)ui->listViewSelectedSamples->model();
    QStandardItemModel *model = (QStandardItemModel*)ui->listViewSelectedGroups->model();

    // Invert selection of the group list
    for(int i = 1; i < (workingDataSet->ngroups+1); i++) {
    if (model->item(i)->checkState()==Qt::Checked)
       model->item(i)->setCheckState(Qt::Unchecked);
    else
        model->item(i)->setCheckState(Qt::Checked);
      }

    /* set the samples of the selected groups as checked */

    for(int i = 0; i < workingDataSet->nsamples; i++) {
        int group = workingDataSet->sample[i].group;
        selSamplesModel->item(i+1)->setCheckState(model->item(group)->checkState());
    }

    /* set all the samples (graphical items) as unselected (to avoid conflicts with the funtion that verifies the checkboxes) */
    for(int i = 0; i < workingDataSet->nsamples; i++) {
        GraphNode* node = hashSampleId2GraphNodeRef[i];
        if(node == NULL){
            continue;
        }
        node->setSelected(false);
    }

    markSamplesAsSelectedByClassOrGroup();
    updateLabelsSelectedClassesGroupsSamples();
    verifyCheckBoxesInListViews();
    updateNumbSamplesShown();


}




void MainWindow::on_comboBoxNodeType_currentTextChanged()
{
    qDebug();

    if(hashSampleId2GraphNodeRef.size() <= 0){
        return;
    }

    /* change the drawing option for each node */
    for(int i = 0; i < workingDataSet->nsamples; i++) {
        GraphNode* node = hashSampleId2GraphNodeRef[i];
        if(node == NULL){
            continue;
        }
        node->drawOption = nodeTypeToDrawOption(ui->comboBoxNodeType->currentText());
        node->mountToolTip();
    }

    /* paint the nodes acording to the chosen node color scheme */
    ui->projGraphicsView->paintNodes();
    projGraphicsScene->update();

    if(ui->comboBoxNodeType->currentText() == "Image") {
        ui->checkBoxShowClassGroupColorOnTooltip->show();
        ui->checkBoxShowClassGroupColorOnTooltip->setChecked(true);
    }
    else {
        ui->checkBoxShowClassGroupColorOnTooltip->hide();
        ui->checkBoxShowClassGroupColorOnTooltip->setChecked(false);
    }
}

void MainWindow::on_checkBoxShowClassGroupColorOnTooltip_stateChanged()
{
    qDebug();

    /* set the flag for each node */
    for(int i = 0; i < workingDataSet->nsamples; i++) {
        GraphNode* node = hashSampleId2GraphNodeRef[i];
        if(node == NULL){
            continue;
        }
        node->showClassGroupColorOnTooltip = ui->checkBoxShowClassGroupColorOnTooltip->isChecked();
    }

    projGraphicsScene->update();
}

void MainWindow::on_comboBoxNodeColor_currentTextChanged()
{
    qDebug();

    /* paint the nodes acording to the chosen node color scheme */
    currentNodeColorOption = ui->comboBoxNodeColor->currentText();
    markSamplesAsSelectedByClassOrGroup();
    updateSampleTooltip();
    ui->projGraphicsView->paintNodes();
    projGraphicsScene->update();

}

void MainWindow::on_comboBoxFilterClass_currentIndexChanged()
{
    qDebug();

    filterSamplesByClassGroupFeaturesAndNodeData();
}

void MainWindow::on_comboBoxFilterGroup_currentIndexChanged()
{
    qDebug();

    filterSamplesByClassGroupFeaturesAndNodeData();
}

void MainWindow::filterSamplesByClassGroupFeaturesAndNodeData()
{
    qDebug();

    if(workingDataSet == NULL)
        return;

    if(hashSampleId2GraphNodeRef.isEmpty())
        return;

    /* filter the samples by their trueclass/group */
    int filterClass = ui->comboBoxFilterClass->currentIndex();
    int filterGroup = ui->comboBoxFilterGroup->currentIndex();
    bool inverseClassFilter = ui->checkBoxInverseClassFilter->isChecked();
    bool inverseGroupFilter = ui->checkBoxInverseGroupFilter->isChecked();

    if(filterGroup == 0 && filterClass == 0) {
        for(int i = 0; i < workingDataSet->nsamples; ++i) {
            GraphNode* node = hashSampleId2GraphNodeRef[i];
            if(node == NULL)
                continue;
            if(inverseClassFilter || inverseGroupFilter)
                node->setVisible(false);
            else
                node->setVisible(true);
        }
    } else if(filterGroup == 0 && filterClass != 0) {
        for (int i = 0; i < workingDataSet->nsamples; ++i) {
            GraphNode* node = hashSampleId2GraphNodeRef[i];
            if(node == NULL)
                continue;

            bool rptaClass;
            if(inverseClassFilter)
                rptaClass = workingDataSet->sample[i].truelabel != filterClass;
            else
                rptaClass = workingDataSet->sample[i].truelabel == filterClass;

            if(rptaClass)
                node->setVisible(true);
            else
                node->setVisible(false);
        }
    } else if(filterGroup != 0 && filterClass == 0) {
        for(int i = 0; i < workingDataSet->nsamples; ++i) {
            GraphNode* node = hashSampleId2GraphNodeRef[i];
            if(node == NULL)
                continue;

            bool rptaGroup;
            if(inverseGroupFilter)
                rptaGroup = workingDataSet->sample[i].group != filterGroup;
            else
                rptaGroup = workingDataSet->sample[i].group == filterGroup;

            if(rptaGroup)
                node->setVisible(true);
            else
                node->setVisible(false);

        }
    } else {
        for(int i = 0; i < workingDataSet->nsamples; ++i) {
            GraphNode* node = hashSampleId2GraphNodeRef[i];
            if(node == NULL)
                continue;

            bool rptaClass;
            if(inverseClassFilter)
                rptaClass = workingDataSet->sample[i].truelabel != filterClass;
            else
                rptaClass = workingDataSet->sample[i].truelabel == filterClass;

            bool rptaGroup;
            if(inverseGroupFilter)
                rptaGroup = workingDataSet->sample[i].group != filterGroup;
            else
                rptaGroup = workingDataSet->sample[i].group == filterGroup;

            if(rptaClass && rptaGroup)
                node->setVisible(true);
            else
                node->setVisible(false);
        }
    }

    /* filter the samples by their node data */
    float dataFrom = ui->doubleSpinBoxNodeDataFilterFrom->value();
    float dataTo = ui->doubleSpinBoxNodeDataFilterTo->value();

    for(int i = 0; i < workingDataSet->nsamples; ++i) {
        GraphNode* node = hashSampleId2GraphNodeRef[i];
        if(node == NULL)
            continue;

        if(ui->comboBoxNodeDataFilter->currentText() == "Perplexity") {
            if(!(node->featPerplexity >= dataFrom && node->featPerplexity <= dataTo))
                node->setVisible(false);
        }
        else if(ui->comboBoxNodeDataFilter->currentText() == "Mean") {
                if(!(node->featMean >= dataFrom && node->featMean <= dataTo))
                    node->setVisible(false);
        }
        else if(ui->comboBoxNodeDataFilter->currentText() == "Stdev") {
                    if(!(node->featStdev >= dataFrom && node->featStdev <= dataTo))
                        node->setVisible(false);
        }
        else if(ui->comboBoxNodeDataFilter->currentText() == "Weight") {
                    if(!(node->featWeight >= dataFrom && node->featWeight <= dataTo))
                        node->setVisible(false);
        }
    }

    /* filter the samples by their features */
    float featuresFrom = ui->doubleSpinBoxFeaturesFrom->value();
    float featuresTo = ui->doubleSpinBoxFeaturesTo->value();

    for(int i = 0; i < workingDataSet->nsamples; ++i) {
        GraphNode* node = hashSampleId2GraphNodeRef[i];
        if(node == NULL)
            continue;
        if(!(iftMinFloatArray(node->samplePtr->feat, workingDataSet->nfeats) >= featuresFrom &&
             iftMaxFloatArray(node->samplePtr->feat, workingDataSet->nfeats) <= featuresTo))
            node->setVisible(false);
    }

    updateNumbSamplesShown();
    projGraphicsScene->update();
}

void MainWindow::timer_slot()
{
    qDebug();

    /* print the corresponding message and start the timer again */
    QString text;
    if(threadTaskType == THREAD_TSNE_PROJECTION) {
        text = QString("Creating projection with t-SNE ... (%1)").arg(formatTime(threadElapsedTime.elapsed()));
    }

    ui->statusBar->showMessage(text);
    threadTimer->start(1000);
}

QString MainWindow::formatTime(int miliseconds)
{
    qDebug();

    int secs = miliseconds / 1000;
    int mins = (secs / 60) % 60;
    //int hours = (secs / 3600);
    secs = secs % 60;

    return QString("%1:%2").arg(mins, 2, 10, QLatin1Char('0')).arg(secs, 2, 10, QLatin1Char('0'));
}

DrawingOption MainWindow::nodeTypeToDrawOption(QString nodeType)
{
    DrawingOption drawOption;

    if(nodeType == "Point")
        drawOption = optionPoint;
    if(nodeType == "Feature pie")
        drawOption = optionFeatPie;
    if(nodeType == "Feature histogram")
        drawOption = optionFeatHistogram;
    if(nodeType == "Number")
        drawOption = optionText;
    if(nodeType == "Image")
        drawOption = optionImage;
    if(nodeType == "Point and Edge"){
        drawOption = optionDrawingEdges;
    }
    return drawOption;
}

void MainWindow::on_comboBoxSupervisionField_currentTextChanged()
{
    qDebug();

    if(resetPropagation())
        currentSupervisionField = ui->comboBoxSupervisionField->currentText();
    else
        ui->comboBoxSupervisionField->setCurrentText(currentSupervisionField);
}

void MainWindow::on_comboBoxSupervisionMode_currentTextChanged()
{
    qDebug();

    if(resetPropagation())
        currentSupervisionMode = ui->comboBoxSupervisionMode->currentText();
    else
        ui->comboBoxSupervisionMode->setCurrentText(currentSupervisionMode);
}

void MainWindow::on_pushButtonResetSupervision_clicked()
{
    qDebug();

    if(resetPropagation()) {
        ui->comboBoxFilterClass->setCurrentIndex(0);
        ui->comboBoxFilterGroup->setCurrentIndex(0);
    }
}

bool MainWindow::resetPropagation()
{
    qDebug();

    if(QMessageBox::warning(this, tr("Warning"), tr("The propagation will be lost. Do you want to reset?"),QMessageBox::Yes | QMessageBox::No) == QMessageBox::No)
        return false;

    if(workingDataSet != NULL)
        iftDestroyDataSet(&workingDataSet);

    workingDataSet = iftCopyDataSet(originalDataSet, true);
    ui->projGraphicsView->paintNodes();
    projGraphicsScene->update();
    updateNumbSamplesShown();
    updateDataSetInfoTextBoxes();
    createHashBetweenClassesAndColors();
    createComboBoxForClassAndGroupFilters();

    return true;
}


void MainWindow::createGraphEdgesbykNN()
{
    qDebug();

    iftDataSet *Z = iftCopyDataSet(workingDataSet, true);
    iftDataSet *Z2 = NULL;
    iftKnnGraph *knn_graph = NULL;
    int k_max = 10;//(int)ui->horizontalSliderPerplexity->value();

    Z2 = iftExtractSamples(Z,IFT_TRAIN);

    if(Z2->nsamples){
        // Creating knn graph
        knn_graph = iftCreateKnnGraph(Z2, k_max);

        // Creating Edges
        // for each node in knn_graph
        for(int u=0; u<knn_graph->nnodes; u++){
            GraphNode *source = hashSampleId2GraphNodeRef[u];
            source->edgeMode = EdgeMode_Neighbours;
            /*
            double max_arcw_adj = IFT_INFINITY_FLT_NEG;
            iftAdjSet *adj = knn_graph->node[u].adj;
            for(int k=0; k<knn_graph->kmax; k++){
                if(adj->arcw > max_arcw_adj)
                    max_arcw_adj = adj->arcw;
                adj = adj->next;
            }*/

            iftAdjSet *adj = knn_graph->node[u].adj;
            // it creates k edges
            while(adj){
                int id_dest = adj->node;
                GraphEdge *edge = new GraphEdge(source,hashSampleId2GraphNodeRef[id_dest]);

                // it applies alpha for each edge by the its normalized local distance
                double hue = adj->arcw/knn_graph->node[u].maxarcw;
                hue = 1.0 - hue; // close to 1 (bad/red) and 0 (good/green)
                hue = pow(hue,0.2); // transfer function
                hue *= 120;
                edge->fillColor.setHsv(hue, 255, 255, 120);

                adj = adj->next;
                graphEdges.push_back(edge);
            }
        }
    }

    iftDestroyDataSet(&Z);
    iftDestroyKnnGraph(&knn_graph);

    // adding the edges to the scene
    for(int j=0; j<graphEdges.size(); j++){
        //graphEdges.at(j)->setFlag(QGraphicsItem::ItemIgnoresTransformations);
        projGraphicsScene->addItem(graphEdges.at(j));
    }

    safetyInd = new SafetyIndicator();
    safetyInd->setFlag(QGraphicsItem::ItemIgnoresTransformations);
    safetyInd->setZValue(2.0);
    projGraphicsScene->addItem(safetyInd);

}


void MainWindow::clearGraphEdges()
{
    qDebug();

    for(int j=0;j<graphNodes.size(); j++){
        GraphNode *node = graphNodes.at(j);
        if(node->sourceEdges.size()>0){
            node->sourceEdges.clear();
        }
        if(node->destEdges.size()>0){
            node->destEdges.clear();
        }
    }
}

double MainWindow::calcEntropy(QVector<int> freq, int n)
{
    qDebug();

    // it calculates the normalized vector
    QVector<float> freq_norm(freq.size(), 0.0);
    double sum = 0.0;
    for (int i = 1; i <= freq.size(); i++)
        sum += freq[i];
    for (int i = 1; i <= freq.size(); i++)
        freq_norm[i] = freq[i]/sum;

    // it calculates the entropy for the normalized vector
    double logn = log(n);
    double entropy = 0.0;
    for (int i = 1; i <= freq_norm.size(); i++)
          if (freq_norm[i] > 0)
              entropy += freq_norm[i] * (log(freq_norm[i])-logn);
    entropy = (-1.0)*entropy/n;

    return entropy;
}

void MainWindow::createHashBetweenIDAndGlobalkNNColor()
{
    qDebug();

    for(int id_source=0; id_source<workingDataSet->nsamples; id_source++){

        int c = 0; //number of different labels in NN
        int l = workingDataSet->sample[id_source].truelabel;
        GraphNode *node = hashSampleId2GraphNodeRef[id_source];
        int n = node->sourceEdges.size();
        QVector<int> labels(workingDataSet->nclasses+1, 0); // vector for number of adj with diferent labels
        int nUnsup = 0;

        // it calculates "labels" for each label and c
        for(int i=0; i<n; i++){
            GraphNode *adj = node->sourceEdges.at(i)->destinationNode;
            int id_adj = hashGraphNodeRef2SampleId[adj];
            //if the destination node is supervised
            if(iftHasSampleStatus(workingDataSet->sample[id_adj], IFT_SUPERVISED))
                labels[workingDataSet->sample[id_adj].truelabel]++;
            if(!(iftHasSampleStatus(workingDataSet->sample[id_adj], IFT_SUPERVISED)))
                nUnsup++;
        }
        for(int i=1; i<=workingDataSet->nclasses; i++){
            if(labels[i]>0)
                c++;
        }

        printf("---- nUnsup %d", nUnsup);
        fflush(stdout);

        // it calculates the color for the indicator
        if(c==0) // all the adjacents are unsupervised
            hashSampleId2GlobalkNNColor.insert(id_source, QColor::fromHsv(-1,120, 120));// QColor(200,200,200,255)); //gray
        else
        {
            if(c==1){ // there is one label
                // if the source node is supervised
                if(iftHasSampleStatus(workingDataSet->sample[id_source], IFT_SUPERVISED))
                {
                    if(labels[l]>0){ // l=k
                        double s = double(labels[l])/n;
                        s = (int)(s*255);

                        hashSampleId2GlobalkNNColor.insert(id_source, QColor::fromHsv(120,s,255)); //green
                        //if (s == 0 )
                        //qDebug(" c = 1 l= sup %f, %d ", s, n);
                    }
                    else {
                            //red
                            double entropy = calcEntropy(labels, n);
                            double area = 0;

                            for(int i=1; i<=workingDataSet->nclasses; i++)
                                area += labels[i];
                            area = area/n;
                            double red = (area+entropy)/2;
                            red = (int)((1-red)*60);

                            //green
                            int green = (int)((double(labels[l])/n*60))+60;
                            //qDebug(" c = 1 l!=  sup %f %d ", red, green);
                            hashSampleId2GlobalkNNColor.insert(id_source, QColor::fromHsv(red+green,255,255)); //red+green
                            //qDebug("Nao ENtrou ");
                    }
                }
                if(!(iftHasSampleStatus(workingDataSet->sample[id_source], IFT_SUPERVISED))) { // if the source node is unsupervised
                    int label_c=0;
                    for(int i=1; i<=workingDataSet->nclasses; i++)
                        if(labels[i]>0)
                            label_c = i;

                    double s = (double)(labels[label_c])/n;
                    s = (int)(s*255);
                    //qDebug(" c = 1 unsup %f, %d", s, labels[label_c]);
                    hashSampleId2GlobalkNNColor.insert(id_source, QColor::fromHsv(120,s,255)); //green
                }
            }
            else { // c > 1
                // if the source node is supervised
                if(iftHasSampleStatus(workingDataSet->sample[id_source], IFT_SUPERVISED))
                {
                    //red
                    double entropy = calcEntropy(labels, n);
                    double area = 0;

                    for(int i=1; i<=workingDataSet->nclasses; i++)
                        area += labels[i];
                    area = area/n;
                    double red = (area+entropy)/2;
                    red = (int)((1-red)*60);

                    //green
                    int green = (int)((labels[l]/n*60))+60;

                    //qDebug(" c > 1 sup %f, %d ", red, green);

                    hashSampleId2GlobalkNNColor.insert(id_source, QColor::fromHsv(red+green,255,255));//red+green
                }
                if(!(iftHasSampleStatus(workingDataSet->sample[id_source], IFT_SUPERVISED))){
                    // if the source node is unsupervised
                    double entropy = calcEntropy(labels, n);
                    double area = 0;

                    for(int i=0; i<workingDataSet->nclasses; i++)
                        area += labels[i+1];
                    area = area/n;

                    double s = (area+entropy)/2;
                    s = (int)((1-s)*255);
                    //qDebug(" c > 1 unsup %f ", s);
                    hashSampleId2GlobalkNNColor.insert(id_source, QColor::fromHsv(0,s,255));//red
                }
            }
        }
    }

}

void MainWindow::on_pushButtonSaveCanvas_clicked()
{
    qDebug();

    QString canvasImgName = QFileDialog::getSaveFileName(this, tr("Save canvas"), workingDir, tr("PNG files (*.png);; All files (*.*)"));

    if (!canvasImgName.isEmpty()) {
        QRect sceneRect = QRect(ui->projGraphicsView->mapFromScene(ui->projGraphicsView->sceneRect().topLeft()) + QPoint(2, 2), ui->projGraphicsView->mapFromScene(ui->projGraphicsView->sceneRect().bottomRight()));
        QPixmap pixMap = QPixmap::grabWidget(ui->projGraphicsView, sceneRect);
        pixMap.save(canvasImgName);
    }
}

void MainWindow::on_pushButtonChangeBkgColor_clicked()
{
    qDebug();

    QColor color = QColorDialog::getColor(ui->projGraphicsView->backgroundColor, this);
    if( color.isValid() )
    {
      ui->projGraphicsView->backgroundColor = color;

      int h, s, v;
      color.getHsv(&h, &s, &v);

      if(h > 180 || s > 200 || v < 60) {
          ui->projGraphicsView->sceneRectColor = QColor(Qt::white);

          for(int i = 0; i < workingDataSet->nsamples; i++) {
              GraphNode* node = hashSampleId2GraphNodeRef[i];
              if(node == NULL)
                  continue;
              node->borderColor_unselected = QColor(Qt::white);
          }
      }
      else {
          ui->projGraphicsView->sceneRectColor = QColor(Qt::black);

          for(int i = 0; i < workingDataSet->nsamples; i++) {
              GraphNode* node = hashSampleId2GraphNodeRef[i];
              if(node == NULL)
                  continue;
              node->borderColor_unselected = QColor(Qt::black);
          }
      }
    }

    ui->projGraphicsView->resetCachedContent();
    projGraphicsScene->update();
    ui->projGraphicsView->paintNodes();
}

void MainWindow::on_comboBoxNodeDataFilter_currentTextChanged(const QString &arg1)
{
    qDebug();

    /* update the text of the corresponding checkbox */
    if(arg1 == "Perplexity") {
        ui->checkBoxDisplayNodeDataFilter->setText("Display perplexity");
    }
    else if(arg1 == "Mean") {
        ui->checkBoxDisplayNodeDataFilter->setText("Display mean");
    }
    else if(arg1 == "Stdev") {
        ui->checkBoxDisplayNodeDataFilter->setText("Display stdev");
    }
    else if(arg1 == "Weight") {
        ui->checkBoxDisplayNodeDataFilter->setText("Display Weight");
    }

    /* set the values for spinboxes according to the chosen field (the filters are automatically applied when the values are changed) */
    setValuesForNodeDataFilterSpinBoxes();

    /* update node data mode for each sample */
    for (int i = 0; i < workingDataSet->nsamples; ++i) {
        GraphNode* node = hashSampleId2GraphNodeRef[i];
        if(node == NULL){
            continue;
        }
        node->setCurrentNodeDataMode(ui->comboBoxNodeDataFilter->currentText());
    }
}

void MainWindow::on_doubleSpinBoxNodeDataFilterFrom_valueChanged(double arg1)
{
    qDebug();

    ui->doubleSpinBoxNodeDataFilterTo->setMinimum(arg1 + ui->doubleSpinBoxNodeDataFilterTo->singleStep());
    filterSamplesByClassGroupFeaturesAndNodeData();
}

void MainWindow::on_doubleSpinBoxNodeDataFilterTo_valueChanged(double arg1)
{
    qDebug();

    ui->doubleSpinBoxNodeDataFilterFrom->setMaximum(arg1 - ui->doubleSpinBoxNodeDataFilterFrom->singleStep());
    filterSamplesByClassGroupFeaturesAndNodeData();
}

void MainWindow::on_checkBoxAutomaticPerplexity_stateChanged()
{
    qDebug();

    if(ui->checkBoxAutomaticPerplexity->isChecked()) {
        if(originalDataSet) {
            computeAutomaticPerplexity();
        }
        ui->spinBoxPerplexity->setEnabled(false);
        ui->horizontalSliderPerplexity->setEnabled(false);
    } else{
        ui->spinBoxPerplexity->setEnabled(true);
        ui->horizontalSliderPerplexity->setEnabled(true);
    }
}

void MainWindow::on_doubleSpinBoxFeaturesFrom_valueChanged(double arg1)
{
    qDebug();

    ui->doubleSpinBoxFeaturesTo->setMinimum(arg1 + ui->doubleSpinBoxFeaturesTo->singleStep());
    filterSamplesByClassGroupFeaturesAndNodeData();
}

void MainWindow::on_doubleSpinBoxFeaturesTo_valueChanged(double arg1)
{
    qDebug();

    ui->doubleSpinBoxFeaturesFrom->setMaximum(arg1 - ui->doubleSpinBoxFeaturesFrom->singleStep());
    filterSamplesByClassGroupFeaturesAndNodeData();
}


void MainWindow::computeAutomaticPerplexity()
{
    qDebug();

    int *sampPerClass = iftCountSamplesPerClassDataSet(originalDataSet);
    sampPerClass[0] = IFT_INFINITY_INT;
    int smallerClass = iftArgMin(sampPerClass, originalDataSet->nclasses+1);
    ui->spinBoxPerplexity->setMaximum((float)originalDataSet->nsamples/3.0);
    ui->spinBoxPerplexity->setValue(sampPerClass[smallerClass]);
    ui->horizontalSliderPerplexity->setMaximum((float)originalDataSet->nsamples/3.0);
    ui->horizontalSliderPerplexity->setSingleStep((float)ui->horizontalSliderPerplexity->maximum()/10.0);
    ui->horizontalSliderPerplexity->setValue(sampPerClass[smallerClass]);
}

void MainWindow::on_checkBoxDisplayNodeDataFilter_stateChanged()
{
    qDebug();

    for (int i = 0; i < workingDataSet->nsamples; ++i) {
        GraphNode* node = hashSampleId2GraphNodeRef[i];
        node->setDisplayNodeDataAsPointBorder(ui->checkBoxDisplayNodeDataFilter->isChecked());
    }
    projGraphicsScene->update();
}

void MainWindow::on_checkBoxInverseClassFilter_stateChanged()
{
    qDebug();

    filterSamplesByClassGroupFeaturesAndNodeData();
}

void MainWindow::on_checkBoxInverseGroupFilter_stateChanged()
{
    qDebug();

    filterSamplesByClassGroupFeaturesAndNodeData();
}

void MainWindow::on_pushButtonEnableShadow_clicked()
{
    qDebug();

    drawShadowInPoints = !drawShadowInPoints;

    for (int i = 0; i < workingDataSet->nsamples; ++i) {
        GraphNode* node = hashSampleId2GraphNodeRef[i];
        node->setDrawShadowInPoints(drawShadowInPoints);
    }
    ui->projGraphicsView->paintNodes();
}

void MainWindow::on_listViewSelectedClasses_clicked(const QModelIndex &index)
{
    qDebug();

    QStandardItemModel *selClassesModel = (QStandardItemModel*)ui->listViewSelectedClasses->model();

    /* if the item 0 was clicked, all the others get its check state */
    if(index.row() == 0) {
        for(int i = 1; i < workingDataSet->nclasses+1; i++)
            selClassesModel->item(i)->setCheckState(selClassesModel->item(0)->checkState());
    }

    addClassToSelectedClassesListView();
}

void MainWindow::on_listViewSelectedGroups_clicked(const QModelIndex &index)
{
    qDebug();

    QStandardItemModel *selGroupsModel = (QStandardItemModel*)ui->listViewSelectedGroups->model();

    printf("%d\n",hashGroupId2GroupName.size());

    /* if the item 0 was clicked, all the others get its check state */
    if(index.row() == 0) {
        for(int i = 1; i < hashGroupId2GroupName.size(); i++)
            selGroupsModel->item(i)->setCheckState(selGroupsModel->item(0)->checkState());
    }

    addGroupToSelectedGroupsListView();
}

void MainWindow::on_listViewSelectedSamples_clicked(const QModelIndex &index)
{
    qDebug();

    QStandardItemModel *selSamplesModel = (QStandardItemModel*)ui->listViewSelectedSamples->model();

    /* if the item 0 was clicked, all the others get its check state */
    if(index.row() == 0) {
        for(int i = 1; i < workingDataSet->nsamples+1; i++)
            selSamplesModel->item(i)->setCheckState(selSamplesModel->item(0)->checkState());
    }

    addSampleToSelectedSamplesListView();
}


void MainWindow::on_pushButtonSaveSelectedClasses_clicked()
{
    qDebug();

    QString filename = QFileDialog::getSaveFileName(this, tr("Save selected classes"), workingDir, tr("JSON files (*.json);; All files (*.*)"));
    QStandardItemModel *model = (QStandardItemModel*)ui->listViewSelectedClasses->model();

    if (!filename.isEmpty()) {
        /* create a json array with the selected classes */
        QJsonArray selClasses;
        for(int i = 1; i < workingDataSet->nclasses+1; i++)
            if(model->item(i)->checkState() == Qt::Checked)
                selClasses.append(QJsonValue(i));

        /* create a json object and save it */
        QJsonObject jsonObj;
        jsonObj["dataset"] = dataSetName;
        jsonObj["selected_classes"] = selClasses;
        QJsonDocument jsonDoc(jsonObj);

        QFile file(filename);
        if(file.open(QFile::WriteOnly |QFile::Truncate)) {
            QTextStream output(&file);
            output << jsonDoc.toJson(QJsonDocument::Indented);
        }
        file.close();
    }
}

void MainWindow::on_pushButtonSaveSelectedGroups_clicked()
{
    qDebug();

    QString filename = QFileDialog::getSaveFileName(this, tr("Save selected groups"), workingDir, tr("JSON files (*.json);; All files (*.*)"));
    QStandardItemModel *model = (QStandardItemModel*)ui->listViewSelectedGroups->model();

    if (!filename.isEmpty()) {
        /* create a json array with the selected groups */
        QJsonArray selGroups;
        for(int i = 1; i < workingDataSet->ngroups+1; i++)
            if(model->item(i)->checkState() == Qt::Checked)
                selGroups.append(QJsonValue(i));

        /* create a json object and save it */
        QJsonObject jsonObj;
        jsonObj["dataset"] = dataSetName;
        jsonObj["selected_groups"] = selGroups;
        QJsonDocument jsonDoc(jsonObj);

        QFile file(filename);
        if(file.open(QFile::WriteOnly |QFile::Truncate)) {
            QTextStream output(&file);
            output << jsonDoc.toJson(QJsonDocument::Indented);
        }
        file.close();
    }
}

void MainWindow::on_pushButtonSaveSelectedSamples_clicked()
{
    qDebug();

    QString filename = QFileDialog::getSaveFileName(this, tr("Save selected samples"), workingDir, tr("JSON files (*.json);; All files (*.*)"));
    QStandardItemModel *model = (QStandardItemModel*)ui->listViewSelectedSamples->model();

    if (!filename.isEmpty()) {
        /* create a json array with the selected samples */
        QJsonArray selSamples;
        for(int i = 1; i < workingDataSet->nsamples+1; i++)
            if(model->item(i)->checkState() == Qt::Checked)
                selSamples.append(QJsonValue(i-1));

        /* create a json object and save it */
        QJsonObject jsonObj;
        jsonObj["dataset"] = dataSetName;
        jsonObj["selected_samples"] = selSamples;
        QJsonDocument jsonDoc(jsonObj);

        QFile file(filename);
        if(file.open(QFile::WriteOnly |QFile::Truncate)) {
            QTextStream output(&file);
            output << jsonDoc.toJson(QJsonDocument::Indented);
        }
        file.close();
    }
}

void MainWindow::on_pushButtonOpenSelectedClasses_clicked()
{
    qDebug();

    QString filename = QFileDialog::getOpenFileName(this, tr("Open selected classes"), workingDir, tr("JSON files (*.json);; All files (*.*)"));
    QStandardItemModel *model = (QStandardItemModel*)ui->listViewSelectedClasses->model();

    /* read the selected classes from a JSON file */
    if (!filename.isEmpty()) {
        for(int i = 1; i < workingDataSet->nclasses+1; i++)
            model->item(i)->setCheckState(Qt::Unchecked);

        QFile file(filename);
        if(file.open(QFile::ReadOnly | QFile::Truncate)) {
            QJsonDocument jsonDoc= QJsonDocument::fromJson(file.readAll());
            QJsonObject jsonObj = jsonDoc.object();

            QJsonArray selClasses = jsonObj["selected_classes"].toArray();
            for(int i = 0; i < selClasses.size() ; i++) {
                int classId = selClasses[i].toInt();
                if(classId != 0)
                    model->item(classId)->setCheckState(Qt::Checked);
            }
        }
    }
}

void MainWindow::on_pushButtonOpenSelectedGroups_clicked()
{
    qDebug();

    QString filename = QFileDialog::getOpenFileName(this, tr("Open selected groups"), workingDir, tr("JSON files (*.json);; All files (*.*)"));
    QStandardItemModel *model = (QStandardItemModel*)ui->listViewSelectedGroups->model();

    /* read the selected classes from a JSON file */
    if (!filename.isEmpty()) {
        for(int i = 1; i < workingDataSet->ngroups+1; i++)
            model->item(i)->setCheckState(Qt::Unchecked);

        QFile file(filename);
        if(file.open(QFile::ReadOnly | QFile::Truncate)) {
            QJsonDocument jsonDoc= QJsonDocument::fromJson(file.readAll());
            QJsonObject jsonObj = jsonDoc.object();

            QJsonArray selGroups = jsonObj["selected_groups"].toArray();
            for(int i = 0; i < selGroups.size() ; i++) {
                int groupId = selGroups[i].toInt();
                if(groupId != 0)
                    model->item(groupId)->setCheckState(Qt::Checked);
            }
        }
    }
}

void MainWindow::on_pushButtonOpenSelectedSamples_clicked()
{
    qDebug();

    QString filename = QFileDialog::getOpenFileName(this, tr("Open selected samples"), workingDir, tr("JSON files (*.json);; All files (*.*)"));
    QStandardItemModel *model = (QStandardItemModel*)ui->listViewSelectedSamples->model();

    /* read the selected samples from a JSON file */
    if (!filename.isEmpty()) {
        for(int i = 1; i < workingDataSet->nsamples+1; i++)
            model->item(i)->setCheckState(Qt::Unchecked);

        QFile file(filename);
        if(file.open(QFile::ReadOnly | QFile::Truncate)) {
            QJsonDocument jsonDoc= QJsonDocument::fromJson(file.readAll());
            QJsonObject jsonObj = jsonDoc.object();

            QJsonArray selSamples = jsonObj["selected_samples"].toArray();
            for(int i = 0; i < selSamples.size() ; i++) {
                int sampleId = selSamples[i].toInt() + 1;
                if(sampleId != 0)
                    model->item(sampleId)->setCheckState(Qt::Checked);
            }
        }
    }
}

/*//////////////////////////////////////////// MYTHREAD /////////////////////////////////////////////*/
// WARNING: The code that is being executed by thread must not modify graphical components (ui)
// All the processing using the ui pointer (to modify its components) must be performed inside the MainWindow class.
// This also includes calling functions of MainWindows that modify the ui content. They must not be called
/*///////////////////////////////////////////////////////////////////////////////////////////////////*/

MyThread::MyThread(MainWindow* window, ThreadTaskType taskType)
{
    qDebug();

    this->mainWindow = window;
    this->taskType = taskType;
}

void MyThread::run()
{
    qDebug();

    bool tsneExecuted = false;
    if(taskType == THREAD_TSNE_PROJECTION) {
        /* perform t-SNE projection or copy a previous projection (if exists) */
        if(mainWindow->originalDataSet->projection != NULL) {
            mainWindow->workingDataSet = iftCopyDataSet(mainWindow->originalDataSet, true);
            tsneExecuted = false;

        } else {
            double perplexity = (double)mainWindow->ui->horizontalSliderPerplexity->value();
            size_t maxIter = (size_t)mainWindow->ui->horizontalSliderNumIterTSNE->value();
            if(mainWindow->workingDataSet != NULL)
	      iftDestroyDataSet(&(mainWindow->workingDataSet));
            mainWindow->workingDataSet = iftDimReductionByTSNE(mainWindow->originalDataSet, 2, perplexity, maxIter);
            /* we must copy originalDataSet after the projection because it contains both the original and the projected data */
            iftDestroyDataSet(&mainWindow->workingDataSet);
            mainWindow->workingDataSet = iftCopyDataSet(mainWindow->originalDataSet,true);
            tsneExecuted = true;
                qInfo( "C Style Info Message" );


        }
        emit projectionFinished_signal(tsneExecuted);
    } else if (taskType == THREAD_THUMBNAIL) {
        /* Initialization */

        char const *ext = nullptr;
        char output[250];
        iftImage *img = nullptr;
        iftImage *slice = nullptr;
        iftImage *resized = nullptr;

        for (int i = 0; i < mainWindow->fileSet->n; i++){
            if (!QFile::exists(mainWindow->fileSet->files[i]->path)){
                mainWindow->thumbnailPath->files[i] = iftCreateFile("");
                continue;
            }
            ext = iftFileExt(mainWindow->fileSet->files[i]->path);
            if (iftCompareStrings(ext,".mimg")){
                iftMImage *mimg = iftReadMImage(mainWindow->fileSet->files[i]->path);
                img = iftMImageToImage(mimg,255,mimg->m/2);
                iftDestroyMImage(&mimg);
            } else {
                img = iftReadImageByExt(mainWindow->fileSet->files[i]->path);
            }

            sprintf(output,"%s/%s/%s.png","thumbnails",iftFilename(mainWindow->dataSetName.toUtf8().data(),iftFileExt(mainWindow->dataSetName.toUtf8().data())),iftFilename(mainWindow->fileSet->files[i]->path,ext));
            mainWindow->thumbnailPath->files[i] = iftCreateFile(output);
            qDebug() << output;

            if (iftIs3DImage(img)){
                slice = iftGetXYSlice(img,img->zsize/2);
                resized = iftResizeImage(slice,32,32,1);
                iftImage *norm = iftNormalize(resized,0,255);
                iftDestroyImage(&slice);
                iftWriteImageByExt(norm,output);
                iftDestroyImage(&norm);
            } else {
                resized = iftResizeImage(img,32,32,1);
                iftWriteImageByExt(resized,output);
            }
            iftDestroyImage(&img);
            iftDestroyImage(&resized);
        }
        emit thumbnailFinished_signal();
    } else if (taskType == THREAD_GRAPH_EDGES) {
        mainWindow->createGraphEdgesbykNN();
        emit graphEdgesFinished_signal();
    }
}
