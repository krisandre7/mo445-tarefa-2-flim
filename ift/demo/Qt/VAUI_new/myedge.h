#ifndef MYEDGE_H
#define MYEDGE_H

#include <QGraphicsItem>
#include "mynode.h"

typedef enum {
    undirectedEdge = 0,
    directedEdge = 1,
    predecessorEdge = 2
} EdgeType;

class MyNode;

class MyEdge : public QGraphicsItem
{
public:
    MyEdge();
    MyEdge(MyNode* _sourceNode, MyNode* _destNode);
    MyNode* sourceNode;
    MyNode* destinationNode;
    QPointF sourcePoint;
    QPointF destinationPoint;
    qreal arrowSize;
    qreal lineWidth;

    QRectF boundingRect() const;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *, QWidget *);
    void adjust();
};

#endif // MYEDGE_H
