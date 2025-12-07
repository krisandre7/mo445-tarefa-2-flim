#ifndef SAFETYINDICATOR
#define SAFETYINDICATOR

#include <QPainter>
#include <QGraphicsItem>

class SafetyIndicator : public QGraphicsItem
{
public:
    SafetyIndicator();
    ~SafetyIndicator();

    float radiusX;
    float radiusY;

    QRect obj_boundingRect;

    QColor fillColor;
    QColor borderColor;
    qreal lineWidth;

    QColor nodeColor;

    QRectF boundingRect() const;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
};

#endif // SAFETYINDICATOR

