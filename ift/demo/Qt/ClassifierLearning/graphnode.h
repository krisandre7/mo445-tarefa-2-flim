#ifndef GRAPHNODE_H
#define GRAPHNODE_H

#include <QPainter>
#include <QGraphicsItem>
#include <QDebug>
#include <QMouseEvent>
#include <QGraphicsSceneHoverEvent>
#include <QFileInfo>
#include "graphedge.h"
#include "globalvariables.h"
#include "safetyindicator.h"
#include "ift.h"
#include <QGraphicsDropShadowEffect>

typedef enum {
    optionPoint = 0,
    optionFeatPie = 1,
    optionFeatHistogram = 2,
    optionText = 3,
    optionImage = 4,
    optionDrawingEdges = 5
} DrawingOption;

typedef enum {
    EdgeMode_None,
    EdgeMode_Predecessor,
    EdgeMode_Neighbours
} EdgeMode;

class GraphEdge;

class GraphNode : public QGraphicsItem
{
public:
    GraphNode(iftSample *sampPtr, int numFeats);
    ~GraphNode();
    float radiusX;
    float radiusY;
    float radiusX_highlight;
    float radiusY_highlight;
    float radiusX_unhighlight;
    float radiusY_unhighlight;
    iftSample *samplePtr = NULL;
    int nFeats;
    float featPerplexity;
    float featMean;
    float featStdev;
    float featWeight;
    bool displayNodeDataAsPointBorder;
    QString currentNodeDataMode;
    float maxGlobalFeatPerplexity;
    float minGlobalFeatPerplexity;
    float maxGlobalFeatMean;
    float minGlobalFeatMean;
    float maxGlobalFeatStdev;
    float minGlobalFeatStdev;
    float maxGlobalFeatWeight;
    float minGlobalFeatWeight;
    QRect obj_boundingRect;
    QColor fillColor;
    QColor borderColor_unselected;
    QColor borderColor_selected;
    QColor borderColor_supervised;
    QColor borderColor_propagated;
    qreal fontSize;
    QString sampleImagePath;
    QString thumbnailImagePath;
    QString sampleName;
    bool currentSupervising;
    bool isMarked;
    bool isHovered;
    unsigned long numberTimesChecked;
    EdgeMode edgeMode;
    DrawingOption drawOption;
    qreal Z;
    QString sampleText;
    qreal fontSizeScalingFactor;
    QVector<GraphEdge*> sourceEdges;
    QVector<GraphEdge*> destEdges;
    SafetyIndicator *safetyIndic;
    QGraphicsDropShadowEffect *shadowEffect;
    bool drawShadowInPoints;
    bool isSelectedByClassOrGroup;
    bool showClassGroupColorOnTooltip;

    QRectF boundingRect() const;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    void mountToolTip();
    void hoverEnterEvent();
    void hoverLeaveEvent();
    void computeFeatPerplexity();
    void computeFeatMean();
    void computeFeatStdev();
    void computeFeatWeight();
    void setMaxGlobalFeatPerplexity(float val);
    void setMinGlobalFeatPerplexity(float val);
    void setMaxGlobalFeatMean(float val);
    void setMinGlobalFeatMean(float val);
    void setMaxGlobalFeatStdev(float val);
    void setMinGlobalFeatStdev(float val);
    void setMaxGlobalFeatWeight(float val);
    void setMinGlobalFeatWeight(float val);
    void setDisplayNodeDataAsPointBorder(bool val);
    void setCurrentNodeDataMode(QString val);
    void setDrawShadowInPoints(bool val);
};

#endif // GRAPHNODE_H
