#include "mygraphicwidget.h"
#include <QDebug>
#include <QGraphicsItem>
#include <mynode.h>
#include <QMenu>


MyGraphicWidget::MyGraphicWidget(QWidget *parent): QGraphicsView(parent)
{
    setMouseTracking(true);
    start = clock();
    myTimer.start();
    validStart = false;
    autoReprojection = false;
}

void MyGraphicWidget::wheelEvent(QWheelEvent *event)
{
    scaleView(pow((double)2, -event->delta() / 240.0));
}

void MyGraphicWidget::scaleView(qreal scaleFactor)
{
    qreal factor = transform().scale(scaleFactor, scaleFactor).mapRect(QRectF(0, 0, 1, 1)).width();
    if (factor < 0.07 || factor > 100)
        return;

    scale(scaleFactor, scaleFactor);
}

void MyGraphicWidget::drawBackground(QPainter *painter, const QRectF &rect)
{
    Q_UNUSED(rect);

    QColor background(235,235,235);

    // Shadow
    QRectF sceneRect = this->sceneRect();
    QRectF rightShadow(sceneRect.right(), sceneRect.top() + 5, 5, sceneRect.height());
    QRectF bottomShadow(sceneRect.left() + 5, sceneRect.bottom(), sceneRect.width(), 5);
    if (rightShadow.intersects(rect) || rightShadow.contains(rect))
        painter->fillRect(rightShadow, background);
    if (bottomShadow.intersects(rect) || bottomShadow.contains(rect))
        painter->fillRect(bottomShadow, background);

    // Fill
    QLinearGradient gradient(sceneRect.topLeft(), sceneRect.bottomRight());
    //gradient.setColorAt(0, Qt::white);
    gradient.setColorAt(0, background);
    gradient.setColorAt(1, background);
    painter->fillRect(rect.intersected(sceneRect), gradient);
    painter->setBrush(Qt::NoBrush);
    painter->drawRect(sceneRect);

}

void MyGraphicWidget::mouseMoveEvent(QMouseEvent *event)
{
    //    printf("miau %d %d\n",event->pos().y(),event->pos().x());
    //    fflush(stdout);
    nMilliseconds = myTimer.elapsed();
    MouseX_new = event->pos().x();
    MouseY_new = event->pos().y();
    QGraphicsItem * gItem2 =itemAt(MouseX_old,MouseY_old);
    if(gItem2 != NULL){
        qDebug() << "entrei " << event->pos().x() << event->pos().y() << counter ;
        counter++;
    }
    if(nMilliseconds > totalTimeReaction){
        QGraphicsItem * gItem =itemAt(MouseX_old,MouseY_old);
        if(gItem != NULL){
            if(gItem != lastItem){
                MyNode* node = dynamic_cast<MyNode*>(gItem);
                node->numberTimesChecked += 1;
                lastItem = gItem;
            }
        }
    }
    myTimer.restart();
    MouseX_old = MouseX_new;
    MouseY_old = MouseY_new;
    QGraphicsView::mouseMoveEvent(event);
}

void MyGraphicWidget::mousePressEvent(QMouseEvent *event){

    if (event->button() == Qt::RightButton) {
        event->accept();
        return;
    }
    QGraphicsView::mousePressEvent(event);
}

void MyGraphicWidget::contextMenuEvent(QContextMenuEvent *event)
{
    emit myMouseContextMenuSignal(event);
}

void MyGraphicWidget::updateSceneAutoReprojection(bool autoReproject){
    autoReprojection = autoReproject;
}

void MyGraphicWidget::resizeEvent(QResizeEvent* event){

    if(autoReprojection == true){
        int minDimension = (event->size().height() < event->size().width())? event->size().height() : event->size().width();
        scene()->setSceneRect(0,0,minDimension,minDimension);
        emit sceneAreaSizeChanged();
    }
    QGraphicsView::resizeEvent(event);
}

//void MyGraphicWidget::mousePressEvent(QMouseEvent *event)
//{
//    QGraphicsView::mousePressEvent(event);
//}





//void MyGraphicWidget::mousePressEvent(QMouseEvent *event)
//{
//    QPointF mousePositionInGraphicArea = event->pos();
//    QPointF mousePositionInScene = mapToScene(event->pos());
//    if (event->button() == Qt::RightButton) {
//        event->accept();
//        return;
//    }
//    QGraphicsView::mousePressEvent(event);
//}
