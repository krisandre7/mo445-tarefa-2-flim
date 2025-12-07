#include "graphnode.h"
#include "mainwindow.h"

GraphNode::GraphNode(iftSample *sampPtr, int numFeats)
{
    radiusX_highlight = pointDefaultRx*3;
    radiusY_highlight = pointDefaultRy*3;
    radiusX_unhighlight = pointDefaultRx;
    radiusY_unhighlight = pointDefaultRy;

    radiusX = radiusX_unhighlight;
    radiusY = radiusY_unhighlight;

    obj_boundingRect.setX(-radiusX);
    obj_boundingRect.setY(-radiusY);
    obj_boundingRect.setWidth(2*radiusX);
    obj_boundingRect.setHeight(2*radiusY);

    samplePtr = sampPtr;
    nFeats = numFeats;

    fillColor = QColor(255,0,0,180);
    borderColor_unselected = QColor(Qt::black);
    borderColor_selected = QColor(Qt::cyan);
    sampleImagePath = "";
    thumbnailImagePath = "";
    sampleName = "";
    Z = -1;
    drawOption = optionPoint;
    fontSizeScalingFactor = 2;
    borderColor_supervised = QColor(Qt::magenta);
    borderColor_propagated = QColor(Qt::yellow);
    fontSize = 0.8;
    isMarked = false;
    edgeMode = EdgeMode_None;
    numberTimesChecked = 0;

    isHovered = false;
    setAcceptHoverEvents(true);
    shadowEffect = new QGraphicsDropShadowEffect();
    shadowEffect->setBlurRadius(50);
    shadowEffect->setOffset(4);
    setGraphicsEffect(shadowEffect);
    drawShadowInPoints = false;
    isSelectedByClassOrGroup = false;

    computeFeatPerplexity();
    computeFeatMean();
    computeFeatStdev();
    computeFeatWeight();
}

GraphNode::~GraphNode()
{
    return;
}

void GraphNode::computeFeatPerplexity()
{
    float entr = 0.0;
    bool valid = true;
    for (int i = 0; i < nFeats && valid; i++)
    {
        float x = samplePtr->feat[i];
        if (!iftAlmostZero(x) && x > 0)
            entr += log2(x);
        else {
            valid = false;
        }
    }

    if(valid)
        featPerplexity = pow(2, -entr/nFeats);
    else
        featPerplexity = -1;

}

void GraphNode::computeFeatMean()
{
    featMean = iftMean(samplePtr->feat, nFeats);
}

void GraphNode::computeFeatStdev()
{
    featStdev = iftStd(samplePtr->feat, nFeats);
}

void GraphNode::computeFeatWeight()
{
    featWeight = samplePtr->weight;
}

QRectF GraphNode::boundingRect() const
{
    return obj_boundingRect;
}

void GraphNode::mountToolTip()
{
    QString text;
    /* add the image (if it is defined) */
    if(!thumbnailImagePath.isEmpty()) {
        if(!sampleName.isEmpty())
            text = "<html><img height=\"60\" src="+thumbnailImagePath+" /><br/>" +sampleName + "</html>";
        else
            text = "<html><img height=\"60\" src="+thumbnailImagePath+" /></html>";
    } else {
        /* add the sampleName, the features and the extra info */
        text = "-> " + sampleName;
        text += "\n-> perplexity: " + (featPerplexity != -1 ? QString::number(featPerplexity, 'f', 4) : "-1.00 (not defined)");
        text += "\n-> weight: " + (featWeight != -1 ? QString::number(featWeight, 'f', 4) : "-1.00 (not defined)");
        text += "\n-> mean: " + QString::number(featMean, 'f', 4);
        text += "\n-> stdev: " + QString::number(featStdev, 'f', 4);
        text += "\n-> feats: ";
        int nMaxFeats = 20;

        if(nFeats < nMaxFeats) {
            for(int f = 0; f < nFeats; f++) {
                text += QString::number(samplePtr->feat[f], 'f', 2);
                if(f < nFeats-1) {
                    if((f+1) % 5 == 0)
                        text += "\n    ";
                    else
                        text += ", ";
                }
            }
        }
        else {
            for(int f = 0; f < nMaxFeats; f++) {
                text += QString::number(samplePtr->feat[f], 'f', 2);
                if(f < nMaxFeats-1)
                    if((f+1) % 5 == 0)
                        text += "\n    ";
                    else
                        text += ", ";
                else
                    text += ", ...";
            }
        }
    }

    setToolTip(text);
}

void GraphNode::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(option);
    Q_UNUSED(widget);
    qreal diameterX = 2*radiusX;
    qreal diameterY = 2*radiusY;
    int alphaValue = 200;

    if(isSelectedByClassOrGroup)
        alphaValue = 20;

    if(drawOption == optionPoint || drawOption == optionDrawingEdges){

        QRadialGradient gradient(-radiusX*0.3, -radiusY*0.3, radiusX);
        QPen pen;

        setZValue(Z);
        fillColor.setAlpha(alphaValue);
        borderColor_selected.setAlpha(alphaValue);
        borderColor_unselected.setAlpha(alphaValue);
        borderColor_supervised.setAlpha(alphaValue);
        borderColor_propagated.setAlpha(alphaValue);
        //if (option->state & QStyle::State_Sunken) {

        if(isSelected()){
            pen = QPen(borderColor_selected, 0.8);
            gradient.setCenter(radiusX*0.3, radiusY*0.3);
            gradient.setFocalPoint(radiusX*0.3, radiusY*0.3);
            gradient.setColorAt(0, fillColor.light(60));
            gradient.setColorAt(1, fillColor.light(120));
            pen.setColor(pen.color().light(120));
        } else {
            if(iftHasSampleStatus(*samplePtr, IFT_SUPERVISED)) {
                pen = QPen(borderColor_supervised, 0.8);
            }
            else {
                if(iftHasSampleStatus(*samplePtr, IFT_LABELPROPAGATED))
                    pen = QPen(borderColor_propagated, 0.8);
                else
                    pen = QPen(borderColor_unselected, 0.8);
            }
            gradient.setCenter(radiusX*0.3, radiusY*0.3);
            gradient.setFocalPoint(radiusX*0.3, radiusY*0.3);
            //gradient.setColorAt(0, QColor(0,0,0,255));
            gradient.setColorAt(0, fillColor.light(20));
            gradient.setColorAt(1, fillColor.light(80));

            if(isHovered){
                if(drawOption == optionDrawingEdges && sourceEdges.size()>0 && edgeMode==EdgeMode_Neighbours){
                    for(int i=0; i<sourceEdges.size(); i++){
                        sourceEdges.at(i)->setVisible(true);
                        sourceEdges.at(i)->adjust();
                    }
                }
            } else{
                if(drawOption == optionDrawingEdges && sourceEdges.size()>0 && edgeMode==EdgeMode_Neighbours){
                    for(int i=0; i<sourceEdges.size(); i++){
                        sourceEdges.at(i)->setVisible(false);
                        sourceEdges.at(i)->adjust();
                    }
                }
            }
        }
        QBrush brush(fillColor);
        painter->setBrush(brush);
        //painter->setBrush(gradient);
        painter->setPen(pen);
        painter->drawEllipse(-radiusX, -radiusY, diameterX, diameterY);
    }
    else if(drawOption == optionFeatPie){
        QPen pen;
        setZValue(Z);
        borderColor_selected.setAlpha(alphaValue);
        borderColor_unselected.setAlpha(alphaValue);
        borderColor_supervised.setAlpha(alphaValue);
        borderColor_propagated.setAlpha(alphaValue);
        if(isSelected()){
            pen = QPen(borderColor_selected, 0.3);
            pen.setColor(pen.color().light(120));
        } else {
            if(iftHasSampleStatus(*samplePtr, IFT_SUPERVISED)) {
                pen = QPen(borderColor_supervised, 0.3);
            }
            else {
                if(iftHasSampleStatus(*samplePtr, IFT_LABELPROPAGATED))
                    pen = QPen(borderColor_propagated, 0.3);
                else
                    pen = QPen(borderColor_unselected, 0.3);
            }
        }
        painter->setPen(pen);

        /* compute the values for the pie (the pie is actually draw if the number of features is not so high and there are no negative features) */
        if(nFeats <= 15 && iftMinFloatArray(samplePtr->feat, nFeats) >= 0) {
            float normVal = iftSumFloatArray(samplePtr->feat, nFeats);
            float startAngle = 0, endAngle = 0, spanAngle = 0;
            QRectF rectangle(-radiusX, -radiusY, diameterX, diameterY);
            iftColorTable *ctb = iftCategoricalColorTable(nFeats);

            for(int f=0; f < nFeats; f++) {
                startAngle = endAngle;
                spanAngle = samplePtr->feat[f] / normVal * 360.0 * 16.0;
                endAngle += spanAngle;
                iftColor YCbCr = ctb->color[f];
                iftColor RGB = iftYCbCrtoRGB(YCbCr, 255);
                QColor color; color.setRed(RGB.val[0]); color.setGreen(RGB.val[1]); color.setBlue(RGB.val[2]);
                color.setAlpha(alphaValue);
                QBrush brush(color);
                painter->setBrush(brush);
                painter->drawPie(rectangle, startAngle, spanAngle);
            }
        }
        else {
            QBrush brush(fillColor);
            painter->setBrush(brush);
            painter->drawEllipse(-radiusX, -radiusY, diameterX, diameterY);
        }
    }
    else if(drawOption == optionFeatHistogram){
        QPen pen;
        setZValue(Z);
        borderColor_selected.setAlpha(alphaValue);
        borderColor_unselected.setAlpha(alphaValue);
        borderColor_supervised.setAlpha(alphaValue);
        borderColor_propagated.setAlpha(alphaValue);
        if(isSelected()){
            pen = QPen(borderColor_selected, 0.3);
            pen.setColor(pen.color().light(120));
        } else {
            if(iftHasSampleStatus(*samplePtr, IFT_SUPERVISED)) {
                pen = QPen(borderColor_supervised, 0.3);
            }
            else {
                if(iftHasSampleStatus(*samplePtr, IFT_LABELPROPAGATED))
                    pen = QPen(borderColor_propagated, 0.3);
                else
                    pen = QPen(borderColor_unselected, 0.3);
            }
        }
        painter->setPen(pen);

        /* draw the histogram only if the number of features is not so high */
        if(nFeats <= 15) {
            /* obtain the normalization value for the histogram (we must use the absolute values of the features to avoid problems with negative values) */
            float *featsAbsVal = iftAllocFloatArray(nFeats);
            for(int f=0; f < nFeats; f++)
                featsAbsVal[f] = fabsf(samplePtr->feat[f]);
            float normVal = iftMaxFloatArray(featsAbsVal, nFeats);

            QSizeF barSize;
            barSize.setHeight(0);
            barSize.setWidth(diameterX / (float)nFeats);
            QPointF upperLeftCorner(-radiusX - barSize.width(), 0);
            iftColorTable *ctb = iftCategoricalColorTable(nFeats);

            /* draw the bars for each feature */
            for(int f=0; f < nFeats; f++) {
                barSize.setHeight(featsAbsVal[f] / normVal * diameterY);
                upperLeftCorner.setX(upperLeftCorner.x() + barSize.width());
                if(samplePtr->feat[f] >= 0)
                    // upperLeftCorner.setY(-radiusY + (diameterY - barSize.height()));
                    upperLeftCorner.setY(-barSize.height());
                else
                    upperLeftCorner.setY(0);
                iftColor YCbCr = ctb->color[f];
                iftColor RGB = iftYCbCrtoRGB(YCbCr, 255);
                QColor color; color.setRgb(RGB.val[0], RGB.val[1], RGB.val[2]);
                color.setAlpha(alphaValue);
                QBrush brush(color);
                painter->setBrush(brush);
                painter->drawRect(QRectF(upperLeftCorner, barSize));
            }
            painter->drawLine(-radiusX - barSize.width(), 0, radiusX + barSize.width(), 0);
        }
        else {
            QBrush brush(fillColor);
            painter->setBrush(brush);
            painter->drawEllipse(-radiusX, -radiusY, diameterX, diameterY);
        }
    }
    else if(drawOption == optionText) {
        QPen pen;
        QFont font = painter->font();
        int boundBoxCeilFactor = 0;
        //font.setPointSize(font.pointSize() / fontSizeScalingFactor);
        //font.setStretch(QFont::UltraCondensed);
        if(isSelected()){
            font.setBold(true);
            font.setItalic(true);
            font.setUnderline(true);
            //font.setStrikeOut(true);
            fillColor.setAlpha(alphaValue);
            borderColor_selected.setAlpha(alphaValue);
            borderColor_unselected.setAlpha(alphaValue);
            borderColor_supervised.setAlpha(alphaValue);
            borderColor_propagated.setAlpha(alphaValue);
            if(edgeMode != EdgeMode_None){
                if(edgeMode == EdgeMode_Predecessor)
                    if(sourceEdges.size() > 0)
                        for (int i = 0; i < sourceEdges.size(); ++i) {
                            GraphEdge* edge = sourceEdges.at(i);
                            if(edge->sourceNode == this) {
                                edge->setVisible(true);
                                edge->destinationNode->setSelected(true);
                                edge->destinationNode->update();
                            }
                        }
                if(edgeMode == EdgeMode_Neighbours)
                    if(sourceEdges.size() > 0)
                        for (int i = 0; i < sourceEdges.size(); ++i)
                            sourceEdges.at(i)->setVisible(true);
            }

        } else {
            font.setBold(false);
            font.setItalic(false);
            font.setUnderline(false);
            if(edgeMode != EdgeMode_None){
                fillColor.setAlphaF(0.2);
                borderColor_selected.setAlpha(alphaValue);
                borderColor_unselected.setAlpha(alphaValue);
                borderColor_supervised.setAlpha(alphaValue);
                borderColor_propagated.setAlpha(alphaValue);
                if(edgeMode == EdgeMode_Predecessor)
                    for (int i = 0; i < sourceEdges.size(); ++i)
                        sourceEdges.at(i)->setVisible(false);
                if(edgeMode == EdgeMode_Neighbours)
                    for (int i = 0; i < sourceEdges.size(); ++i)
                        if(sourceEdges.at(i)->sourceNode->isSelected() != true && sourceEdges.at(i)->destinationNode->isSelected() != true)
                            sourceEdges.at(i)->setVisible(false);
//                for (int i = 0; i < Edges.size(); ++i) {
//                    MyEdge* edge = Edges.at(i);
//                    if(edge->sourceNode->isSelected() != true && edge->destinationNode->isSelected() != true)
//                        edge->setVisible(false);
//                }
            } else {
                fillColor.setAlphaF(alphaValue);
                borderColor_selected.setAlpha(alphaValue);
                borderColor_unselected.setAlpha(alphaValue);
                borderColor_supervised.setAlpha(alphaValue);
                borderColor_propagated.setAlpha(alphaValue);
            }
        }

        if(isMarked == true){
            pen = QPen(QColor(0,0,0,alphaValue), 0);
            painter->setPen(pen);
            painter->drawEllipse(-radiusX, -radiusY, diameterX, diameterY);
        }

        if(iftHasSampleStatus(*samplePtr, IFT_SUPERVISED))
            font.setStrikeOut(true);
        else
            font.setStrikeOut(false);

        font.setPointSize(diameterX/1.3);
        //font.setLetterSpacing(QFont::AbsoluteSpacing,-5);
        painter->setFont(font);
        pen = QPen(fillColor, fontSize);
        painter->setPen(pen);
        QRect boundingRect;
        painter->drawText(-radiusX, -radiusY, diameterX, diameterY, Qt::AlignCenter, sampleText, &boundingRect);
        obj_boundingRect = boundingRect.adjusted(0, 0, -pen.width(), -pen.width());
        boundBoxCeilFactor = radiusY/3.1;
        obj_boundingRect.setY(obj_boundingRect.y()+boundBoxCeilFactor);
        obj_boundingRect.setHeight(obj_boundingRect.height()-boundBoxCeilFactor);
        //painter->drawRect(obj_boundingRect);
    }
    else if(drawOption == optionImage) {
        QImage tmpImage(sampleImagePath);
        if(tmpImage.isNull())
            return;

        painter->drawImage(boundingRect(), tmpImage);

        if(showClassGroupColorOnTooltip) {
            QPen pen = QPen(fillColor, 2.0);
            painter->setBrush(Qt::NoBrush);
            painter->setPen(pen);
            painter->drawRect(boundingRect());
        }
    }

    if(displayNodeDataAsPointBorder) {
        if(drawOption == optionPoint || drawOption == optionFeatPie || drawOption == optionText) {
            QColor color(Qt::black);

            float val=1.0;
            if(currentNodeDataMode == "Perplexity") {
                val = 1.0 - (featPerplexity - minGlobalFeatPerplexity) / (maxGlobalFeatPerplexity - minGlobalFeatPerplexity);
            }
            else if(currentNodeDataMode == "Mean") {
                val = 1.0 - (featMean - minGlobalFeatMean) / (maxGlobalFeatMean - minGlobalFeatMean);
            }
            else if(currentNodeDataMode == "Stdev") {
                val = 1.0 - (featStdev - minGlobalFeatStdev) / (maxGlobalFeatStdev - minGlobalFeatStdev);
            }
            else if(currentNodeDataMode == "Weight") {
                val = 1.0 - (featWeight - minGlobalFeatWeight) / (maxGlobalFeatWeight - minGlobalFeatWeight);
            }

            qreal h,s,v;
            color.getHsvF(&h, &s, &v);
            color.setHsvF(h, s, val);
            color.setAlpha(alphaValue);
            QPen pen = QPen(color, 2.0);
            painter->setPen(pen);
            painter->setBrush(Qt::NoBrush);
            painter->drawEllipse(-radiusX, -radiusY, diameterX, diameterY);
        }
    }

    graphicsEffect()->setEnabled(drawShadowInPoints);
}

void GraphNode::hoverEnterEvent(){
    isHovered=true;
    update();
}

void GraphNode::hoverLeaveEvent(){
    isHovered=false;
    update();
}


void GraphNode::setMaxGlobalFeatPerplexity(float val)
{
    maxGlobalFeatPerplexity = val;
}

void GraphNode::setMinGlobalFeatPerplexity(float val)
{
    minGlobalFeatPerplexity = val;
}

void GraphNode::setMaxGlobalFeatMean(float val)
{
    maxGlobalFeatMean = val;
}

void GraphNode::setMinGlobalFeatMean(float val)
{
    minGlobalFeatMean = val;
}

void GraphNode::setMaxGlobalFeatStdev(float val)
{
    maxGlobalFeatStdev = val;
}

void GraphNode::setMinGlobalFeatStdev(float val)
{
    minGlobalFeatStdev = val;
}


void GraphNode::setMaxGlobalFeatWeight(float val)
{
    maxGlobalFeatWeight = val;
}

void GraphNode::setMinGlobalFeatWeight(float val)
{
    minGlobalFeatWeight= val;
}


void GraphNode::setDisplayNodeDataAsPointBorder(bool val)
{
    displayNodeDataAsPointBorder = val;
}

void GraphNode::setCurrentNodeDataMode(QString val)
{
    currentNodeDataMode = val;
}

void GraphNode::setDrawShadowInPoints(bool val)
{
    drawShadowInPoints = val;
}
