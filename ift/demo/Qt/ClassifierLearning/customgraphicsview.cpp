#include "customgraphicsview.h"
#include "graphnode.h"

CustomGraphicsView::CustomGraphicsView(QWidget *parent): QGraphicsView(parent)
{
    qDebug();

    setMouseTracking(true);
    myTimer.start();
    validStart = false;
    autoReprojection = false;
    mainWindow = dynamic_cast<MainWindow*>(this->window());
    backgroundColor = QColor(Qt::white);
    sceneRectColor = QColor(Qt::black);

    connect(this, SIGNAL(mouseContextMenuRequest_signal(QContextMenuEvent *)), this, SLOT(createMouseMenuForPropagation_slot(QContextMenuEvent *)) );
    connect(this, SIGNAL(sceneAreaSizeChanged_signal()), this, SLOT(updateNodesPosition_slot()) );
}

CustomGraphicsView::~CustomGraphicsView()
{
    qDebug();

    return;
}

void CustomGraphicsView::createMouseMenuForPropagation_slot(QContextMenuEvent *event)
{
    qDebug();

    /* create a popup combo box with the dataset's classes and their colors */
    Q_UNUSED(event);

    /* count the number of selected samples */
    int selectedItems = 0;
    for (int i = 0; i < mainWindow->hashSampleId2GraphNodeRef.size(); i++) {
        GraphNode* node = mainWindow->hashSampleId2GraphNodeRef[i];
        if(node == NULL)
            continue;
        if(node->isSelected())
            selectedItems++;
    }

    /* create the combobox menu */
    QMenu *contextMenu = new QMenu(this);
    if(selectedItems == 0) {
        if(mainWindow->currentSupervisionField == "True label") {
            QAction *actionMenu = contextMenu->addAction(tr("Add new class ..."));
            connect(actionMenu, SIGNAL(triggered(bool)), this, SLOT(addNewClassToDataSet_slot()));
        }
        else if(mainWindow->currentSupervisionField == "Group") {
            QAction *menuAction = contextMenu->addAction(tr("Add new group ..."));
            connect(menuAction, SIGNAL(triggered(bool)), this, SLOT(addNewGroupToDataSet_slot()));
        }
    } else {
        if(mainWindow->currentSupervisionField == "True label") {
            /* menu for label propagation */
            QMenu *contextSubMenu1 = new QMenu(tr("Propagate label"));
            for (int j = 0; j < mainWindow->hashLabelId2LabelName.size(); j++) {
                QPixmap pixmap(10,10);
                pixmap.fill(mainWindow->hashLabelId2LabelColor[j]);
                QIcon icon(pixmap);
                QAction *subMenuAction = contextSubMenu1->addAction(icon, mainWindow->hashLabelId2LabelName[j]);
                subMenuAction->setData(QString::number(j));
                connect(subMenuAction, &QAction::triggered, [this, subMenuAction]() {
                    propagateLabel(subMenuAction->data().toInt());
                });
            }
            contextMenu->addMenu(contextSubMenu1);

            /* menu for sample/class/group selection */
            QMenu *contextSubMenu2 = new QMenu(tr("Select"));
            QAction *subMenuAction1 = contextSubMenu2->addAction(tr("Sample"));
            connect(subMenuAction1, SIGNAL(triggered(bool)), SLOT(selectSample_slot()));
            QAction *subMenuAction2 = contextSubMenu2->addAction(tr("Class"));
            connect(subMenuAction2, SIGNAL(triggered(bool)), SLOT(selectClass_slot()));
            QAction *subMenuAction3 = contextSubMenu2->addAction(tr("Group"));
            connect(subMenuAction3, SIGNAL(triggered(bool)), SLOT(selectGroup_slot()));
            contextMenu->addMenu(contextSubMenu2);

            /* menu for class creation */
            QAction *menuAction = contextMenu->addAction(tr("Add new class ..."));
            connect(menuAction, SIGNAL(triggered(bool)), this, SLOT(addNewClassToDataSet_slot()));
        }
        else if(mainWindow->currentSupervisionField == "Group") {
            /* menu for label propagation */
            QMenu *contextSubMenu1 = new QMenu(tr("Propagate label"));
            for (int j = 0; j < mainWindow->hashGroupId2GroupName.size(); j++) {
                QPixmap pixmap(10,10);
                pixmap.fill(mainWindow->hashGroupId2GroupColor[j]);
                QIcon icon(pixmap);
                QAction *subMenuAction = contextSubMenu1->addAction(icon, mainWindow->hashGroupId2GroupName[j]);
                subMenuAction->setData(QString::number(j));
                connect(subMenuAction, &QAction::triggered, [this, subMenuAction]() {
                    propagateLabel(subMenuAction->data().toInt());
                });
            }
            contextMenu->addMenu(contextSubMenu1);

            /* menu for sample/class/group selection */
            QMenu *contextSubMenu2 = new QMenu(tr("Select"));
            QAction *subMenuAction1 = contextSubMenu2->addAction(tr("Sample"));
            connect(subMenuAction1, SIGNAL(triggered(bool)), SLOT(selectSample_slot()));
            QAction *subMenuAction2 = contextSubMenu2->addAction(tr("Class"));
            connect(subMenuAction2, SIGNAL(triggered(bool)), SLOT(selectClass_slot()));
            QAction *subMenuAction3 = contextSubMenu2->addAction(tr("Group"));
            connect(subMenuAction3, SIGNAL(triggered(bool)), SLOT(selectGroup_slot()));
            contextMenu->addMenu(contextSubMenu2);

            /* menu for group creation */
            QAction *menuAction = contextMenu->addAction(tr("Add new group ..."));
            connect(menuAction, SIGNAL(triggered(bool)), this, SLOT(addNewGroupToDataSet_slot()));
        }
    }
    contextMenu->popup(QWidget::mapToGlobal(QPoint(event->x(), event->y())));
}

void CustomGraphicsView::propagateLabel(int labelIdPropagated)
{
    qDebug();

    iftDataSet *dataset = mainWindow->workingDataSet;

    if(dataset == NULL || mainWindow->hashSampleId2GraphNodeRef.isEmpty())
        return;

    if(mainWindow->currentSupervisionField == "True label") {
        /* change the truelabel in the selected samples */
        for (int i = 0; i < dataset->nsamples; i++) {
            GraphNode* node = mainWindow->hashSampleId2GraphNodeRef[i];
            if(node == NULL)
                continue;

            if(node->isSelected()) {
                if(!iftHasSampleStatus(dataset->sample[i], IFT_SUPERVISED)) {
                    dataset->sample[i].truelabel = labelIdPropagated;

                    if(labelIdPropagated != 0)
                        iftAddSampleStatus(&dataset->sample[i], IFT_LABELPROPAGATED);
                    else
                        iftRemoveSampleStatus(&dataset->sample[i], IFT_LABELPROPAGATED);
                }
            }
        }
    } else if(mainWindow->currentSupervisionField == "Group") {
        /* change the group in the selected samples */
        for (int i = 0; i < dataset->nsamples; i++) {
            GraphNode* node = mainWindow->hashSampleId2GraphNodeRef[i];
            if(node == NULL)
                continue;

            if(node->isSelected()) {
                if(!iftHasSampleStatus(dataset->sample[i], IFT_SUPERVISED)) {
                    dataset->sample[i].group = labelIdPropagated;

                    if(labelIdPropagated != 0)
                        iftAddSampleStatus(&dataset->sample[i], IFT_LABELPROPAGATED);
                    else
                        iftRemoveSampleStatus(&dataset->sample[i], IFT_LABELPROPAGATED);
                }
            }
        }
    }
    paintNodes();
}

void CustomGraphicsView::addNewClassToDataSet_slot()
{
    qDebug();

    mainWindow->addNewClassForSupervision();
}

void CustomGraphicsView::addNewGroupToDataSet_slot()
{
    qDebug();

    mainWindow->addNewGroupForSupervision();
}

void CustomGraphicsView::selectClass_slot()
{
    qDebug();

    mainWindow->addClassToSelectedClassesListView();
}

void CustomGraphicsView::selectGroup_slot()
{
    qDebug();

    mainWindow->addGroupToSelectedGroupsListView();
}

void CustomGraphicsView::selectSample_slot()
{
    qDebug();

    mainWindow->addSampleToSelectedSamplesListView();
}

void CustomGraphicsView::updateNodesPosition_slot()
{
    qDebug();

    iftDataSet *dataset = mainWindow->workingDataSet;
    if(dataset == NULL || mainWindow->graphNodes.size() <= 0 || mainWindow->projGraphicsScene == NULL)
        return;

    /* recompute the scene's scaling factor and the position of the nodes */
    computeSceneScalingFactor();

    qreal rx = pointDefaultRx;
    qreal ry = pointDefaultRy;
    qreal factor = mainWindow->sceneScalingFactor - rx*4;

    for (int i = 0; i < mainWindow->graphNodes.size(); i++) {
        qreal x = (dataset->projection->val[i*2])*factor + rx*2;
        qreal y = (dataset->projection->val[i*2 + 1])*factor + ry*2;
        GraphNode* node = mainWindow->graphNodes.at(i);
        node->setPos(x, y);
        node->update();
    }
}

void CustomGraphicsView::computeSceneScalingFactor()
{
    qDebug();

    float rx = pointDefaultRx;
    float ry = pointDefaultRy;
    float scalingFactor_width = (this->size().width()-(rx*5));
    float scalingFactor_height = (this->size().height()-(ry*5));

    if(scalingFactor_height < scalingFactor_width)
        mainWindow->sceneScalingFactor = scalingFactor_height + ry*4;
    else
        mainWindow->sceneScalingFactor = scalingFactor_width + rx*4;


}

void CustomGraphicsView::wheelEvent(QWheelEvent *event)
{
    qDebug();

    scaleView(pow((double)2, -event->delta() / 240.0));
}

void CustomGraphicsView::scaleView(qreal scaleFactor)
{
    qDebug();

    qreal factor = transform().scale(scaleFactor, scaleFactor).mapRect(QRectF(0, 0, 1, 1)).width();
    if (factor < 0.95 || factor > 10)
        return;

    scale(scaleFactor, scaleFactor);
}

void CustomGraphicsView::drawBackground(QPainter *painter, const QRectF &rect)
{
    qDebug();

    Q_UNUSED(rect);

    /* draw a white background */
    QRectF sceneRect = this->sceneRect();
    QLinearGradient gradient(sceneRect.topLeft(), sceneRect.bottomRight());
    gradient.setColorAt(0, backgroundColor);
    gradient.setColorAt(1, backgroundColor);
    painter->fillRect(rect.intersected(sceneRect), gradient);
    painter->setBrush(Qt::NoBrush);

    painter->setPen(sceneRectColor);
    painter->drawRect(sceneRect);
}

void CustomGraphicsView::mouseMoveEvent(QMouseEvent *event)
{
    nMilliseconds = myTimer.elapsed();
    MouseX_new = event->pos().x();
    MouseY_new = event->pos().y();

    QList<QGraphicsItem *> gItems = items(MouseX_old, MouseY_old);
    if(gItems.size()>0)
        for(int index=gItems.size()-1; index>=0; index--){ // it returns the last item
            GraphNode* node = dynamic_cast<GraphNode*>(gItems.at(index));
            if(node){
                if(gItems.at(index) != NULL && gItems.at(index) != lastItem) {
                    /* if the mouse pointer is in the same place during a given time */
                    if(nMilliseconds>totalTimeReaction){
                        node->numberTimesChecked += 1;
                        lastItem = gItems.at(index);
                    }
                }
                /*it shows the safety indicator for the option "Point and Edges" */
                if(mainWindow->safetyInd && node->drawOption == optionDrawingEdges){
                    QPointF p = gItems.at(index)->pos();
                    mainWindow->safetyInd->setPos(p.x()-(node->radiusX/2),
                                                  p.y()-(node->radiusY/2));
                    QColor safetyColor = mainWindow->hashSampleId2GlobalkNNColor[mainWindow->hashGraphNodeRef2SampleId[node]];
                    mainWindow->safetyInd->fillColor = safetyColor;
                    mainWindow->safetyInd->nodeColor = node->fillColor;
                    mainWindow->safetyInd->setVisible(true);
                }
            }
        }
    else
        mainWindow->safetyInd->setVisible(false);

    myTimer.restart();
    MouseX_old = MouseX_new;
    MouseY_old = MouseY_new;
    QGraphicsView::mouseMoveEvent(event);
}

void CustomGraphicsView::mousePressEvent(QMouseEvent *event)
{
    qDebug();

    if (event->button() == Qt::RightButton) {
        event->accept();
        return;
    }
    QGraphicsView::mousePressEvent(event);

}

void CustomGraphicsView::mouseReleaseEvent(QMouseEvent *event)
{
    qDebug();

    mainWindow->updateNumbSamplesShown();
    QGraphicsView::mouseReleaseEvent(event);
}

void CustomGraphicsView::mouseDoubleClickEvent(QMouseEvent *event)
{
    qDebug();

    if (event->button() == Qt::LeftButton) {
        QGraphicsItem * gItem =itemAt(event->pos().x(), event->pos().y());
        if(gItem != NULL) {
            GraphNode* node = dynamic_cast<GraphNode*>(gItem);

            /* create the supervision dialog */
            int id = mainWindow->hashGraphNodeRef2SampleId[node];
            mainWindow->supervisionWindow = new SupervisionWindow(mainWindow, node->sampleImagePath, mainWindow->workingDataSet->sample[id].truelabel, &mainWindow->hashLabelId2LabelColor, &mainWindow->hashLabelId2LabelName);
            mainWindow->supervisionWindow->exec();

            /* mark the sample as supervised */
            if(mainWindow->supervisionWindow->result() == QDialog::Accepted) {
                int labelIdSupervised = mainWindow->supervisionWindow->chosenClass;
                mainWindow->workingDataSet->sample[id].truelabel = labelIdSupervised;

                iftRemoveSampleStatus(&mainWindow->workingDataSet->sample[id], IFT_LABELPROPAGATED);
                if(labelIdSupervised != 0)
                    iftAddSampleStatus(&mainWindow->workingDataSet->sample[id], IFT_SUPERVISED);
                else
                    iftRemoveSampleStatus(&mainWindow->workingDataSet->sample[id], IFT_SUPERVISED);
                paintNodes();
            }
        }
    }
}

void CustomGraphicsView::contextMenuEvent(QContextMenuEvent *event)
{
    qDebug();

    emit mouseContextMenuRequest_signal(event);
}

void CustomGraphicsView::keyPressEvent(QKeyEvent *event)
{
    qDebug();

    // it allows the change of drawing (kNN based)
    QKeyEvent *keyEvent = static_cast<QKeyEvent *>(event);
    if(keyEvent->key() == Qt::Key_M)
        paintNodesByGlobalkNN();
    if(keyEvent->key() == Qt::Key_O) {
        paintNodes();
        iftDataSet *dataset = mainWindow->workingDataSet;
        for (int i = 0; i < dataset->nsamples; i++) {
            GraphNode* node = mainWindow->hashSampleId2GraphNodeRef[i];
            //unsupervised
            if(!(iftHasSampleStatus(dataset->sample[i], IFT_SUPERVISED))){
                node->Z = 0.0;
                node->setZValue(0.0);
            }
            //supervised
            if(iftHasSampleStatus(dataset->sample[i], IFT_SUPERVISED)){
                node->Z = 1.0;
                node->setZValue(1.0);
            }
        }
    }
    // it allows to navigate and select a group number using keyboard
    if(keyEvent->key() == Qt::Key_Space)
        mainWindow->addGroupToSelectedGroupsListView();
    if(keyEvent->key() == Qt::Key_Right)
        mainWindow->SelectGroupUp();
    if(keyEvent->key() == Qt::Key_Left)
        mainWindow->SelectGroupDown();

    
    // it allows the zoom in and out by using
    if(keyEvent->key() == Qt::Key_Up)
        scaleView(pow((double)2, +0.5));
    if(keyEvent->key() == Qt::Key_Down)
        scaleView(pow((double)2, -0.5));

    QWidget::keyPressEvent(event);
}

void CustomGraphicsView::updateSceneAutoReprojection_slot(bool autoReproject)
{
    qDebug();

    autoReprojection = autoReproject;
}

void CustomGraphicsView::resizeEvent(QResizeEvent* event)
{
    qDebug();

    /* update the scene's size */
    if(autoReprojection == true){
        int minDimension = (event->size().height() < event->size().width())? event->size().height() : event->size().width();
        scene()->setSceneRect(0,0,minDimension,minDimension);
        emit sceneAreaSizeChanged_signal();
    }

    QGraphicsView::resizeEvent(event);
}

void CustomGraphicsView::paintNodes()
{
    qDebug();

    /* paint the nodes in the scene accoding to the chosen option */
    if(mainWindow->currentNodeColorOption == "True class"){
        paintNodesByTrueClass();
    }else if(mainWindow->currentNodeColorOption == "Predicted"){
        paintNodesByPredictedClass();
    }else if(mainWindow->currentNodeColorOption == "Supervised"){
        paintNodesBySupervisedClass();
    }else if(mainWindow->currentNodeColorOption == "Propagated"){
        paintNodesByPropagatedClass();
    }else if(mainWindow->currentNodeColorOption == "Supervised + propagated"){
        paintNodesBySupervisedAndPropagatedClass();
    }else if(mainWindow->currentNodeColorOption == "Group"){
        paintNodesByGroup();
    }else if(mainWindow->currentNodeColorOption == "Weight"){
        paintNodesByWeight();
    }

    mainWindow->projGraphicsScene->update();

}

void CustomGraphicsView::paintNodesByTrueClass()
{
    qDebug();

    if(mainWindow->hashSampleId2GraphNodeRef.size() <= 0)
        return;

    iftDataSet *dataset = mainWindow->workingDataSet;
    for (int i = 0; i < dataset->nsamples; i++) {
        GraphNode* node = mainWindow->hashSampleId2GraphNodeRef[i];
        if(node == NULL)
            continue;

        if(iftHasSampleStatus(dataset->sample[i], IFT_SUPERVISED)) {
            QColor color = mainWindow->hashLabelId2LabelColor[dataset->sample[i].truelabel];
            node->fillColor = color;
            node->sampleText = QString("%1").arg(dataset->sample[i].truelabel);
        } else {
            QColor color(0,0,0,255);
            node->fillColor = color;
            node->sampleText = "";
        }
        node->isMarked = false;
    }
}

void CustomGraphicsView::paintNodesByPredictedClass()
{
    qDebug();

    if(mainWindow->hashSampleId2GraphNodeRef.size() <= 0)
        return;

    iftDataSet *dataset = mainWindow->workingDataSet;
    for (int i = 0; i < dataset->nsamples; i++) {
        GraphNode* node = mainWindow->hashSampleId2GraphNodeRef[i];
        if(node == NULL)
            continue;
        QColor color = mainWindow->hashLabelId2LabelColor[dataset->sample[i].label];
        node->fillColor = color;
        node->sampleText = QString("%1").arg(dataset->sample[i].label);
        node->isMarked = false;
    }
}

void CustomGraphicsView::paintNodesBySupervisedClass()
{
    qDebug();

    if(mainWindow->hashSampleId2GraphNodeRef.size() <= 0 )
        return;

    iftDataSet *dataset = mainWindow->workingDataSet;
    for (int i = 0; i < dataset->nsamples; i++) {
        GraphNode* node = mainWindow->hashSampleId2GraphNodeRef[i];
        if(node == NULL)
            continue;
        if(iftHasSampleStatus(dataset->sample[i], IFT_SUPERVISED)) {
            QColor color = mainWindow->hashLabelId2LabelColor[dataset->sample[i].truelabel];
            node->fillColor = color;
            node->sampleText = QString("%1").arg(dataset->sample[i].truelabel);
        } else {
            QColor color(0,0,0,255);
            node->fillColor = color;
            node->sampleText = "";
        }
        node->isMarked = false;
    }
}

void CustomGraphicsView::paintNodesByPropagatedClass()
{
    qDebug();

    if(mainWindow->hashSampleId2GraphNodeRef.size() <= 0 )
        return;

    iftDataSet *dataset = mainWindow->workingDataSet;
    for (int i = 0; i < dataset->nsamples; i++) {
        GraphNode* node = mainWindow->hashSampleId2GraphNodeRef[i];
        if(node == NULL)
            continue;
        if(iftHasSampleStatus(dataset->sample[i], IFT_LABELPROPAGATED)) {
            QColor color = mainWindow->hashLabelId2LabelColor[dataset->sample[i].truelabel];
            node->fillColor = color;
            node->sampleText = QString("%1").arg(dataset->sample[i].truelabel);
        } else {
            QColor color(0,0,0,255);
            node->fillColor = color;
            node->sampleText = "";
        }
        node->isMarked = false;
    }
}

void CustomGraphicsView::paintNodesBySupervisedAndPropagatedClass()
{
    qDebug();

    if(mainWindow->hashSampleId2GraphNodeRef.size() <= 0 )
        return;

    iftDataSet *dataset = mainWindow->workingDataSet;
    for (int i = 0; i < dataset->nsamples; i++) {
        GraphNode* node = mainWindow->hashSampleId2GraphNodeRef[i];
        if(node == NULL)
            continue;
        if(iftHasSampleStatus(dataset->sample[i], IFT_SUPERVISED) || iftHasSampleStatus(dataset->sample[i], IFT_LABELPROPAGATED)) {
            QColor color = mainWindow->hashLabelId2LabelColor[dataset->sample[i].truelabel];
            node->fillColor = color;
            node->sampleText = QString("%1").arg(dataset->sample[i].truelabel);
        } else {
            QColor color(0,0,0,255);
            node->fillColor = color;
            node->sampleText = "";
        }
        node->isMarked = false;
    }
}

void CustomGraphicsView::paintNodesByGroup()
{
    qDebug();

    if(mainWindow->hashSampleId2GraphNodeRef.size() <= 0)
        return;

    iftDataSet *dataset = mainWindow->workingDataSet;
    for (int i = 0; i < dataset->nsamples; i++) {
        GraphNode* node = mainWindow->hashSampleId2GraphNodeRef[i];
        if(node == NULL)
            continue;
        QColor color = mainWindow->hashGroupId2GroupColor[dataset->sample[i].group];
        node->fillColor = color;
        node->sampleText = QString("%1").arg(dataset->sample[i].group);
        node->isMarked = false;
    }
}



void CustomGraphicsView::paintNodesByWeight()
{
    qDebug();

    if(mainWindow->hashSampleId2GraphNodeRef.size() <= 0 )
        return;

    iftDataSet *dataset = mainWindow->workingDataSet;
    iftColorTable *ctb = NULL, *ctb2 = NULL;
    ctb2 = iftRedToBlueColorTable(256);
    ctb  = iftConvertYCbCrColorTableToRGBColorTable(ctb2, 255);

    for (int i = 0; i < dataset->nsamples; ++i) {
        GraphNode* node = mainWindow->hashSampleId2GraphNodeRef[i];
        if(node == NULL)
            continue;

	int idx = (dataset->sample[i].weight)*255;
	QColor color = QColor(ctb->color[idx].val[0], ctb->color[idx].val[1],ctb->color[idx].val[2]);
	node->fillColor = color;
	node->sampleText = QString("%1").arg(dataset->sample[i].truelabel);
        node->isMarked = false;
    }
    iftDestroyColorTable(&ctb);
    iftDestroyColorTable(&ctb2);
}



void CustomGraphicsView::paintNodesByGlobalkNN()
{
    qDebug();

    if(mainWindow->hashSampleId2GraphNodeRef.size() <= 0)
        return;

    iftDataSet *dataset = mainWindow->workingDataSet;
    for (int i = 0; i < dataset->nsamples; i++) {
        GraphNode* node = mainWindow->hashSampleId2GraphNodeRef[i];
        if(node == NULL)
            continue;
        QColor color = mainWindow->hashSampleId2GlobalkNNColor[dataset->sample[i].id];
        node->fillColor = color;
        node->sampleText = "";
        node->isMarked = false;

        /*it defines the order for painting*/
        if(color.hue() < 0 || color.hue()>120){ // gray or > green
            node->Z = 0.0;
            node->setZValue(0.0);
        }
        if(color.hue() > 90 && color.hue() <= 120){ //green more yellow
            node->Z = 0.4;
            node->setZValue(0.4);
        }
        if(color.hue() > 60 && color.hue() <= 90){ //green
            node->Z = 0.6;
            node->setZValue(0.6);
        }
        if(color.hue() > 30 && color.hue() <= 60){ // red more yellow
            node->Z = 0.8;
            node->setZValue(0.8);
        }
        if(color.hue() >= 0 && color.hue() <= 30){ // more red
            node->Z = 1.0;
            node->setZValue(1.0);
        }
    }
    mainWindow->projGraphicsScene->update();
}

/*
void CustomGraphicsView::paintNodesByLocalkNN()
{
    qDebug();

    if(mainWindow->hashSampleId2GraphNodeRef.size() <= 0)
        return;

    iftDataSet *dataset = mainWindow->workingDataSet;
    for (int i = 0; i < dataset->nsamples; i++) {
        GraphNode* node = mainWindow->hashSampleId2GraphNodeRef[i];
        if(node == NULL)
            continue;
        QColor color = mainWindow->hashSampleId2kNNColor[dataset->sample[i].id];
        node->fillColor = color;
        node->sampleText = "";
        node->isMarked = false;

        //it defines the order for painting
        if(color.hue() < 0 || color.hue()>120){ // gray or > green
            node->Z = 0.0;
            node->setZValue(0.0);
        }
        if(color.hue() > 60 && color.hue() <= 120){ //green
            node->Z = 0.5;
            node->setZValue(0.5);
        }
        if(color.hue() >= 0 && color.hue() <= 60){ // red
            node->Z = 1.0;
            node->setZValue(1.0);
        }
    }
    mainWindow->projGraphicsScene->update();
}*/

