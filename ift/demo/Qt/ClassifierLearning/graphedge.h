#ifndef GRAPHEDGE_H
#define GRAPHEDGE_H

#include <QGraphicsItem>
#include "graphnode.h"

typedef enum {
    undirectedEdge = 0,
    directedEdge = 1,
    predecessorEdge = 2
} EdgeType;

class GraphNode;

class GraphEdge : public QGraphicsItem
{
public:
    GraphEdge();
    GraphEdge(GraphNode* _sourceNode, GraphNode* _destNode);
    ~GraphEdge();
    GraphNode* sourceNode = NULL;
    GraphNode* destinationNode = NULL;
    QPointF sourcePoint;
    QPointF destinationPoint;
    qreal arrowSize;
    qreal lineWidth;
    QColor fillColor;
    QColor borderColor;

    QRectF boundingRect() const;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *, QWidget *);
    void adjust();

};

#endif // GRAPHEDGE_H
