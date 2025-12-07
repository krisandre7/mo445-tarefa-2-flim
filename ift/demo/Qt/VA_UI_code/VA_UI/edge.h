#ifndef EDGE_H
#define EDGE_H


#include <QGraphicsItem>

typedef enum {
    undirectedEdge = 0,
    directedEdge = 1,
    predecessorEdge = 2
} EdgeType;

class Node;

class Edge : public QGraphicsItem
{
public:
    Edge(Node *sourceNode, Node *destNode);

    Node *sourceNode() const;
    Node *destNode() const;

    void adjust();
    void setEdgeType(EdgeType type);
    EdgeType getEdgeType();


protected:
    QRectF boundingRect() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;

private:
    Node *source, *dest;
    EdgeType edgeType;
    QPointF sourcePoint;
    QPointF destPoint;
    qreal arrowSize;

};

#endif // EDGE_H
