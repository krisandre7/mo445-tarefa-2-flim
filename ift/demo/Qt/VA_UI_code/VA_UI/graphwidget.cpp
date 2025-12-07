#include "graphwidget.h"
#include "edge.h"
#include "node.h"

#include <math.h>

#include <QKeyEvent>
#include <QGraphicsSceneMouseEvent>
#include <QtGui>

GraphWidget::GraphWidget(QWidget *parent)
    : QGraphicsView(parent), timerId(0)
{
    QGraphicsScene *scene = new QGraphicsScene(this);
    scene->setItemIndexMethod(QGraphicsScene::NoIndex);

    //scene->setSceneRect(0, 0, 400, 400);
    setScene(scene);
    setCacheMode(CacheBackground);
    setViewportUpdateMode(BoundingRectViewportUpdate);
    setRenderHint(QPainter::Antialiasing);
    setTransformationAnchor(AnchorUnderMouse);
    scale(qreal(1), qreal(1));
    setDragMode(QGraphicsView::RubberBandDrag);

    //setMinimumSize(600, 600);
    //setWindowTitle(tr("Elastic Nodes"));
    actions = createActions();
}

void GraphWidget::itemMoved()
{
    if (!timerId)
        timerId = startTimer(1000 / 25);
}

void GraphWidget::keyPressEvent(QKeyEvent *event)
{
    switch (event->key()) {
    //    case Qt::Key_Up:
    //        centerNode->moveBy(0, -20);
    //        break;
    //    case Qt::Key_Down:
    //        centerNode->moveBy(0, 20);
    //        break;
    //    case Qt::Key_Left:
    //        centerNode->moveBy(-20, 0);
    //        break;
    //    case Qt::Key_Right:
    //        centerNode->moveBy(20, 0);
    //        break;
    case Qt::Key_Plus:
        zoomIn();
        break;
    case Qt::Key_Minus:
        zoomOut();
        break;
        //    case Qt::Key_Space:
        //    case Qt::Key_Enter:
        //        shuffle();
        //        break;
    default:
        QGraphicsView::keyPressEvent(event);
    }
}

void GraphWidget::timerEvent(QTimerEvent *event)
{
    Q_UNUSED(event);

    QList<Node *> nodes;
    foreach (QGraphicsItem *item, scene()->items()) {
        if (Node *node = qgraphicsitem_cast<Node *>(item))
            nodes << node;
    }

    foreach (Node *node, nodes)
        node->calculateForces();

    bool itemsMoved = false;
    foreach (Node *node, nodes) {
        if (node->advance())
            itemsMoved = true;
    }

    if (!itemsMoved) {
        killTimer(timerId);
        timerId = 0;
    }
}

#ifndef QT_NO_WHEELEVENT
void GraphWidget::wheelEvent(QWheelEvent *event)
{
    scaleView(pow((double)2, -event->delta() / 240.0));
}
#endif

void GraphWidget::drawBackground(QPainter *painter, const QRectF &rect)
{
    Q_UNUSED(rect);

    // Shadow
    QRectF sceneRect = this->sceneRect();
    QRectF rightShadow(sceneRect.right(), sceneRect.top() + 5, 5, sceneRect.height());
    QRectF bottomShadow(sceneRect.left() + 5, sceneRect.bottom(), sceneRect.width(), 5);
    if (rightShadow.intersects(rect) || rightShadow.contains(rect))
        painter->fillRect(rightShadow, Qt::darkGray);
    if (bottomShadow.intersects(rect) || bottomShadow.contains(rect))
        painter->fillRect(bottomShadow, Qt::darkGray);

    // Fill
    QLinearGradient gradient(sceneRect.topLeft(), sceneRect.bottomRight());
    //gradient.setColorAt(0, Qt::white);
    gradient.setColorAt(0, Qt::lightGray);
    gradient.setColorAt(1, Qt::lightGray);
    painter->fillRect(rect.intersected(sceneRect), gradient);
    painter->setBrush(Qt::NoBrush);
    painter->drawRect(sceneRect);

}

void GraphWidget::scaleView(qreal scaleFactor)
{
    qreal factor = transform().scale(scaleFactor, scaleFactor).mapRect(QRectF(0, 0, 1, 1)).width();
    if (factor < 0.07 || factor > 100)
        return;

    scale(scaleFactor, scaleFactor);
}



void GraphWidget::zoomIn()
{
    scaleView(qreal(1.2));
}

void GraphWidget::zoomOut()
{
    scaleView(1 / qreal(1.2));
}

void GraphWidget::contextMenuEvent(QContextMenuEvent *event)
{


    QMenu *mainmenu = createMenu(*actions);


    QAction* selectedAction = mainmenu->exec(event->globalPos());
    DrawingManager drawingManager;
    for (int i = 0; i < actions->size(); ++i) {
        if(selectedAction == actions->at(i)){
            bool isChecked = actions->at(i)->isChecked();
            uncheckAllDrawingOption(actions->at(i)->drawingOption);
            actions->at(i)->setChecked(isChecked);
            drawingManager.drawOption = actions->at(i)->drawingOption;
            if(isChecked){
                drawingManager.objectOption = actions->at(i)->objectDrawingOption;
            }else{
                drawingManager.objectOption = NONE;
            }
            break;
        }
    }


    SelectedOptionSginal(drawingManager);
}

void GraphWidget::uncheckAllDrawingOption(DrawingOption _drawingOption){
    for (int i = 0; i < actions->size(); ++i) {
        if(actions->at(i)->drawingOption == _drawingOption){
            actions->at(i)->setChecked(false);
        }
    }
}

QMenu* GraphWidget::createMenu(QList<SceneActionMenu *>actionsList){

    QMenu* mainmenu = new QMenu(this);

    QMenu *node = mainmenu->addMenu("Node");

    //color options
    QMenu *nodeColor = node->addMenu("Color");
    //hidden options
    QMenu *nodeHidden = node->addMenu("Hidden");
    //
    //QMenu *nodeLabelPropagation = node->addMenu("Label propagation");

    for (int i = 0; i < actionsList.size(); ++i) {
        if(actionsList.at(i)->drawingOption == COLOR){
            nodeColor->addAction(actionsList.at(i));
        }else if(actionsList.at(i)->drawingOption == HIDDEN){
            nodeHidden->addAction(actionsList.at(i));
        }else if(actionsList.at(i)->drawingOption == LABEL_PROPAGATION){
            node->addAction(actionsList.at(i));
        }
    }





    return mainmenu;
}

QList<SceneActionMenu *> * GraphWidget::createActions(){
    QList<SceneActionMenu *>* actionList = new QList<SceneActionMenu* >();
    createColorActions(actionList);
    createHiddenActions(actionList);
    createLabelPropagationActions(actionList);
    return actionList;
}

void GraphWidget::createColorActions(QList<SceneActionMenu *>* actionsList){
    SceneActionMenu *colorTrueLabel = new SceneActionMenu("True Label Based",this,
                                                                    COLOR, COLOR_TRUELABELBASED);
    colorTrueLabel->setCheckable(true);
    actionsList->append(colorTrueLabel);

    SceneActionMenu *colorLabel = new SceneActionMenu("Label Based",this,
                                                                    COLOR, COLOR_LABELBASED);
    colorLabel->setCheckable(true);
    actionsList->append(colorLabel);

    SceneActionMenu *colorOPFCluster = new SceneActionMenu("OPF-Cluster Based",this,
                                                                    COLOR, COLOR_OPFCLUSTERBASED);
    colorOPFCluster->setCheckable(true);
    actionsList->append(colorOPFCluster);



}

void GraphWidget::createHiddenActions(QList<SceneActionMenu *>* actionsList){
    SceneActionMenu *hiddenUnsupervised = new SceneActionMenu("Unsupervised",this,
                                                                        HIDDEN, HIDDEN_UNSUPERVISED);
    hiddenUnsupervised->setCheckable(true);
    actionsList->append(hiddenUnsupervised);
}

void GraphWidget::createLabelPropagationActions(QList<SceneActionMenu *>* actionsList){
    SceneActionMenu *labelPropagation = new SceneActionMenu("Label Propagation",this,
                                                                        LABEL_PROPAGATION, LABEL_PROPAGATION_AREA);
    actionsList->append(labelPropagation);
}





void GraphWidget::mouseMoveEvent(QMouseEvent * event){
    QGraphicsView::mouseMoveEvent(event);

    QPointF mousePositionInGraphicArea = event->pos();
    QPointF mousePositionInScene = mapToScene(event->pos());
    emit mouseMoveCoordinates(&mousePositionInGraphicArea,
                                     &mousePositionInScene);
}




void GraphWidget::mousePressEvent(QMouseEvent *event){

    if (event->button() == Qt::RightButton) {
        event->accept();
        return;
    }
    QGraphicsView::mousePressEvent(event);
}
//void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;

