#include "edge.h"
#include "node.h"
#include "graphwidget.h"
#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QPainter>
#include <QStyleOption>

Node::Node(GraphWidget *graphWidget)
    : graph(graphWidget)
{
    //setFlag(ItemIsMovable);
    setFlag(ItemSendsGeometryChanges);
    setCacheMode(DeviceCoordinateCache);
    setZValue(-1);
    diameter = 6;
    radius = diameter/2;
    R = 255;
    G = 0;
    B = 0;
    R_border = 0;
    G_border = 0;
    B_border = 0;
    alpha = 200;
    numberPred = 0;
    selectedByClassifier = false;
}

void Node::addEdge(Edge *edge)
{
    edgeList << edge;
    edge->adjust();
}

QList<Edge *> Node::edges() const
{
    return edgeList;
}

void Node::calculateForces()
{
    if (!scene() || scene()->mouseGrabberItem() == this) {
        newPos = pos();
        return;
    }
    newPos = pos();


    // Sum up all forces pushing this item away
//    qreal xvel = 0;
//    qreal yvel = 0;
//    foreach (QGraphicsItem *item, scene()->items()) {
//        Node *node = qgraphicsitem_cast<Node *>(item);
//        if (!node)
//            continue;

//        QPointF vec = mapToItem(node, 0, 0);
//        qreal dx = vec.x();
//        qreal dy = vec.y();
//        double l = 2.0 * (dx * dx + dy * dy);
//        if (l > 0) {
//            xvel += (dx * 150.0) / l;
//            yvel += (dy * 150.0) / l;
//        }
//    }

//    // Now subtract all forces pulling items together
//    double weight = (edgeList.size() + 1) * 10;
//    foreach (Edge *edge, edgeList) {
//        QPointF vec;
//        if (edge->sourceNode() == this)
//            vec = mapToItem(edge->destNode(), 0, 0);
//        else
//            vec = mapToItem(edge->sourceNode(), 0, 0);
//        xvel -= vec.x() / weight;
//        yvel -= vec.y() / weight;
//    }

//    if (qAbs(xvel) < 0.1 && qAbs(yvel) < 0.1)
//        xvel = yvel = 0;

//    QRectF sceneRect = scene()->sceneRect();
//    newPos = pos() + QPointF(xvel, yvel);
//    newPos.setX(qMin(qMax(newPos.x(), sceneRect.left() + 10), sceneRect.right() - 10));
//    newPos.setY(qMin(qMax(newPos.y(), sceneRect.top() + 10), sceneRect.bottom() - 10));
}

bool Node::advance()
{
    if (newPos == pos())
        return false;

    setPos(newPos);
    return true;
}

QRectF Node::boundingRect() const
{
    qreal adjust = 1;
    qreal dropShadow = 2;

    return QRectF( -radius - adjust, -radius - adjust, diameter + dropShadow + adjust, diameter + dropShadow + adjust);
}

QPainterPath Node::shape() const
{
    QPainterPath path;
    path.addEllipse(-radius, -radius, diameter, diameter);
    return path;
}

void Node::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *)
{
//    painter->setPen(Qt::NoPen);
//    painter->setBrush(Qt::darkGray);
//    painter->drawEllipse(-7, -7, 20, 20);

    QRadialGradient gradient(-radius*0.3, -radius*0.3, radius);
    if (option->state & QStyle::State_Sunken) {
         painter->setPen(painter->pen());
    }
    if(this->isSelected() || selectedByClassifier){
        setZValue(1);
        //edgeList.at(i)->setVisible(true);
        gradient.setCenter(radius*0.3, radius*0.3);
        gradient.setFocalPoint(radius*0.3, radius*0.3);
        gradient.setColorAt(1, QColor(R,G,B,255).light(120));
        gradient.setColorAt(0, QColor(R*0.5,G*0.5,B*0.5,alpha).light(120));
    } else {
        setZValue(-1);
        gradient.setColorAt(0, QColor(R,G,B,alpha));
        gradient.setColorAt(1, QColor(R*0.4,G*0.4,B*0.4,alpha));
    }
    painter->setBrush(gradient);

    //painter->setPen(QPen(Qt::black, 0.5));
    painter->setPen(QPen(QColor(R_border,G_border,B_border), 0.5));
    painter->drawEllipse(-radius, -radius, diameter, diameter);
}

//QVariant Node::itemChange(GraphicsItemChange change, const QVariant &value)
//{
//    switch (change) {
//    case ItemPositionHasChanged:
//        foreach (Edge *edge, edgeList)
//            edge->adjust();
//        graph->itemMoved();
//        break;
//    default:
//        break;
//    };

//    return QGraphicsItem::itemChange(change, value);
//}

void Node::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    update();
    QGraphicsItem::mousePressEvent(event);
}

void Node::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    update();
    QGraphicsItem::mouseReleaseEvent(event);
}

void Node::setRadius(float _radius){
    this->radius = _radius;
    this->diameter = _radius * 2;
}

float Node::Radius(){
    return radius;
}

void Node::setDiameter(float _diameter){
    this->diameter = _diameter;
    this->radius = _diameter/2;
}

void Node::setColor(int _R,int _G,int _B,int _alpha){
    R = _R;
    G = _G;
    B = _B;
    alpha = _alpha;
}

void Node::setColor(int _R,int _G,int _B){
    R = _R;
    G = _G;
    B = _B;
}

void Node::setColorBorder(int _R,int _G,int _B){
    R_border = _R;
    G_border = _G;
    B_border = _B;
}

int Node::getR(){
    return R;
}

int Node::getG(){
    return G;
}
int Node::getB(){
    return B;
}
