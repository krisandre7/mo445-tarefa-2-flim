#include "myedge.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

MyEdge::MyEdge(MyNode* _sourceNode, MyNode* _destNode)
{
    setAcceptedMouseButtons(0);
    sourceNode = _sourceNode;
    destinationNode = _destNode;
    sourceNode->Edges.push_back(this);
    destinationNode->Edges.push_back(this);
    arrowSize = 1;
    lineWidth = 0.1;
    //    edgeType = undirectedEdge;
    adjust();
}

void MyEdge::adjust()
{
    if (!sourceNode || !destinationNode)
        return;

    QLineF line(mapFromItem(sourceNode, 0, 0), mapFromItem(destinationNode, 0, 0));
    qreal length = line.length();

    prepareGeometryChange();

    if (length > qreal(6.)) {
        QPointF edgeOffset((line.dx() * sourceNode->radiusX) / length, (line.dy() * sourceNode->radiusY) / length);
        QPointF edgeOffsetDest((line.dx() * destinationNode->radiusX) / length, (line.dy() * destinationNode->radiusY) / length);
        sourcePoint = line.p1() + edgeOffset;
        destinationPoint = line.p2() - edgeOffsetDest;
    } else {
        sourcePoint = destinationPoint = line.p1();
    }
}

QRectF MyEdge::boundingRect() const
{
    if (!sourceNode || !destinationNode)
        return QRectF();

    qreal penWidth = 0.5;
    qreal extra = (penWidth + arrowSize) / 2.0;

    return QRectF(sourcePoint, QSizeF(destinationPoint.x() - sourcePoint.x(),
                                      destinationPoint.y() - sourcePoint.y()))
            .normalized()
            .adjusted(-extra, -extra, extra, extra);
}

void MyEdge::paint(QPainter *painter, const QStyleOptionGraphicsItem *, QWidget *)
{
    if (sourceNode == NULL || destinationNode == NULL){
        return;
    }
    QLineF line(sourcePoint, destinationPoint);
    if (qFuzzyCompare(line.length(), qreal(0.))){
        return;
    }
    // Draw the line itself
    painter->setPen(QPen(Qt::black, lineWidth, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    painter->drawLine(line);

    // Draw the arrows
    double angle = ::acos(line.dx() / line.length());
    if (line.dy() >= 0){
        angle = (2*M_PI) - angle;
    }
    if(arrowSize > 0){
        QPointF destArrowP1 = destinationPoint + QPointF(sin(angle - M_PI / 3) * arrowSize,
                                                         cos(angle - M_PI / 3) * arrowSize);
        QPointF destArrowP2 = destinationPoint + QPointF(sin(angle - M_PI + M_PI / 3) * arrowSize,
                                                         cos(angle - M_PI + M_PI / 3) * arrowSize);
        painter->setBrush(Qt::black);
        //    painter->drawPolygon(QPolygonF() << line.p1() << sourceArrowP1 << sourceArrowP2);
        painter->drawPolygon(QPolygonF() << line.p2() << destArrowP1 << destArrowP2);
    }
}
