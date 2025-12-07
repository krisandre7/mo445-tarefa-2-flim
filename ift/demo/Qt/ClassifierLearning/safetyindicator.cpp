#include "safetyindicator.h"
#include "globalvariables.h"

SafetyIndicator::SafetyIndicator()
{
    radiusX = 4;
    radiusY = 4;
    fillColor = QColor(200,200,200,255);
    nodeColor = QColor(200,200,200,255);
    borderColor = Qt::black;
    lineWidth = 3;

    obj_boundingRect.setX(-radiusX);
    obj_boundingRect.setY(-radiusY);
    obj_boundingRect.setWidth(2*radiusX);
    obj_boundingRect.setHeight(2*radiusY);

    setVisible(false);
}

SafetyIndicator::~SafetyIndicator()
{
    return;
}


QRectF SafetyIndicator::boundingRect() const
{

    return obj_boundingRect;
}
void SafetyIndicator::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{

    //QRect rect(-radiusX, -radiusY, 2*radiusX, 2*radiusY);
    QRect node(0, 0, pointDefaultRx*2, pointDefaultRy*2);

    QRect rect(-radiusX, -radiusY, 2*(radiusX+pointDefaultRx), 2*(radiusY+pointDefaultRy));

    painter->setBrush(fillColor);
    painter->setRenderHint(QPainter::Antialiasing);
    painter->drawEllipse(rect);

    painter->setBrush(nodeColor);
    painter->setRenderHint(QPainter::Antialiasing);
    painter->drawEllipse(node);
}
