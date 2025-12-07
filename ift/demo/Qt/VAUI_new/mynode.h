#ifndef MYSQUARE_H
#define MYSQUARE_H

#include <QPainter>
#include <QGraphicsItem>
#include <QDebug>
#include <QMouseEvent>
#include <QGraphicsSceneHoverEvent>
#include "myedge.h"
#include "global.h"



//float pointDefaultRx = 6;
//float pointDefaultRy = 6;
//float textDefaultFontSize = pointDefaultRx;

//float boundBoxDefaultX = -textDefaultFontSize;
//float boundBoxDefaultY = -textDefaultFontSize;
//float boundBoxDefaultWidth = 2*textDefaultFontSize;
//float boundBoxDefaultHeight = 2*textDefaultFontSize;

typedef enum {
    optionDrawing = 0,
    optionText = 1,
    optionImage = 2
} DrawingOption;

typedef enum {
    EdgeMode_None,
    EdgeMode_Predecessor,
    EdgeMode_Neighbours
} EdgeMode;

class MyEdge;

class MyNode : public QGraphicsItem
{
public:
    MyNode();
    float radiusX;
    float radiusY;
    float radiusX_highlight;
    float radiusY_highlight;
    float radiusX_unhighlight;
    float radiusY_unhighlight;
    QRect obj_boundingRect;
    QColor fillColor;
    QColor borderColor_unselected;
    QColor borderColor_selected;
    QColor borderColor_supervised;
    qreal fontSize;
    QString sampleImagePath;
    QString sampleName;
    bool currentSupervising;
    bool isSupervised;
    bool isMarked;
    unsigned long numberTimesChecked;
    EdgeMode edgeMode;
    DrawingOption drawOption;
    qreal Z;
    QString text;
    qreal fontSizeScalingFactor;
    QVector<MyEdge*>Edges;
    QRectF boundingRect() const;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    void mountToopTip();


protected:
    //void hoverEnterEvent ( QGraphicsSceneHoverEvent * event );
    //void toolTipEvent();
    //bool eventFilter(QObject *obj, QEvent *event);


};

#endif // MYSQUARE_H
