#ifndef NODE_H
#define NODE_H


#include <QGraphicsItem>
#include <QList>
#include "global.h"

class Edge;
class GraphWidget;
class QGraphicsSceneMouseEvent;

class Node : public QGraphicsItem
{
public:
    Node(GraphWidget *graphWidget);

    void addEdge(Edge *edge);
    QList<Edge *> edges() const;

    enum { Type = UserType + 1 };
    int type() const override { return Type; }

    void calculateForces();
    bool advance();

    QRectF boundingRect() const override;
    QPainterPath shape() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;
    void setRadius(float _radius);
    float Radius();
    void setDiameter(float _diameter);
    void setColor(int _R,int _G,int _B,int _alpha);
    void setColor(int _R,int _G,int _B);
    void setColorBorder(int _R,int _G,int _B);
    int getR();
    int getG();
    int getB();
    int numberPred;
    bool selectedByClassifier;
    //void showPath(Node nodeDestination);
protected:
    //QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;

    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;

private:
    QList<Edge *> edgeList;
    QPointF newPos;
    GraphWidget *graph;
    float diameter;
    float radius;
    int R,G,B, alpha;
    int R_border,G_border,B_border;


};

#endif // NODE_H

