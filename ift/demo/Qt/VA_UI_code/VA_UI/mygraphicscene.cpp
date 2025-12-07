#include "mygraphicscene.h"

MyGraphicScene::MyGraphicScene(QObject *parent) : QGraphicsScene(parent)
{
    firstMouseLocation = QPointF(0,0);
    lastMouseLocation = QPointF(0,0);
}

void MyGraphicScene::mousePressEvent(QGraphicsSceneMouseEvent* ev)
{
    this->firstMouseLocation = ev->scenePos();
}

void MyGraphicScene::mouseReleaseEvent(QGraphicsSceneMouseEvent* ev){
    this->lastMouseLocation = ev->scenePos();
    emit sendMouseCoordinates(firstMouseLocation,lastMouseLocation);
}

void MyGraphicScene::mouseWheelEvent(QWheelEvent* event){
    qDebug() << event->delta();
}

