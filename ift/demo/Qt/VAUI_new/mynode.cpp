#include "mynode.h"
#include <QToolTip>


MyNode::MyNode()
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

    fillColor = QColor(255,0,0,180);
    borderColor_unselected = QColor(0,0,0,255);
    borderColor_selected = QColor(0,255,255,255);
    sampleImagePath = "";
    sampleName = "";
    Z = -1;
    drawOption = optionDrawing;
    fontSizeScalingFactor = 2;
    isSupervised = false;
    borderColor_supervised = QColor(202, 31, 123,255);
    fontSize = 0.8;
    isMarked = false;
    edgeMode = EdgeMode_None;
    numberTimesChecked = 0;
    //setAcceptHoverEvents(true);
}

QRectF MyNode::boundingRect() const
{
    return obj_boundingRect;
}

void MyNode::mountToopTip(){
    QString text_html;
    if(sampleName != "" &&  sampleImagePath != ""){
        //text_html = "<html><img height=\"60\" src="+sampleImagePath+" /><br/>" +sampleName + "</html>"  ;
        text_html = "<html><img height=\"60\" src="+sampleImagePath+" /><br/></html>"  ;
    }
    else if(sampleImagePath == "" && sampleName != ""){
        text_html = sampleName  ;
    }else if(sampleName == "" && sampleImagePath != ""){
        text_html = "<html><img height=\"60\" src="+sampleImagePath+" /></html>"  ;
    }else{
        return;
    }
    setToolTip(text_html);
}

void MyNode::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(option);
    Q_UNUSED(widget);
    qreal diameterX = 2*radiusX;
    qreal diameterY = 2*radiusY;
    if(drawOption == optionDrawing){
        QRadialGradient gradient(-radiusX*0.3, -radiusY*0.3, radiusX);
        QPen pen;
        setZValue(Z);
        //if (option->state & QStyle::State_Sunken) {
        if(this->isSelected()){
            pen = QPen(borderColor_selected, 0.8);
            gradient.setCenter(radiusX*0.3, radiusY*0.3);
            gradient.setFocalPoint(radiusX*0.3, radiusY*0.3);
            gradient.setColorAt(0, fillColor.light(60));
            gradient.setColorAt(1, fillColor.light(120));
            pen.setColor(pen.color().light(120));
        } else {
            if(isSupervised == true){
                pen = QPen(borderColor_supervised, 0.8);
                //pen = QPen(borderColor_unselected, 0.8);
            }else{
                pen = QPen(borderColor_unselected, 0.8);
            }
            gradient.setCenter(radiusX*0.3, radiusY*0.3);
            gradient.setFocalPoint(radiusX*0.3, radiusY*0.3);
            //gradient.setColorAt(0, QColor(0,0,0,255));
            gradient.setColorAt(0, fillColor.light(20));
            gradient.setColorAt(1, fillColor.light(80));
        }
        painter->setBrush(gradient);
        painter->setPen(pen);
        painter->drawEllipse(-radiusX, -radiusY, diameterX, diameterY);
    }else if(drawOption == optionText){
        QPen pen;
        QFont font = painter->font() ;
        int boundBoxCeilFactor = 0;
        //font.setPointSize(font.pointSize() / fontSizeScalingFactor);
        //font.setStretch(QFont::UltraCondensed);
        if(isSelected()){
            font.setBold(true);
            font.setItalic(true);
            font.setUnderline(true);
            //font.setStrikeOut(true);
            //font.setUnderline(true);
            fillColor.setAlphaF(1.0);
            borderColor_selected.setAlphaF(1.0);
            borderColor_supervised.setAlphaF(1.0);
            borderColor_unselected.setAlpha(1.0);
            if(edgeMode != EdgeMode_None){
                if(edgeMode == EdgeMode_Predecessor){
                    if(Edges.size() > 0){
                        for (int i = 0; i < Edges.size(); ++i) {
                            MyEdge* edge = Edges.at(i);
                            if(edge->sourceNode == this){
                                edge->setVisible(true);
                                edge->destinationNode->setSelected(true);
                                edge->destinationNode->update();
                            }
                        }
                    }
                }
                if(edgeMode == EdgeMode_Neighbours){
                    if(Edges.size() > 0){
                        for (int i = 0; i < Edges.size(); ++i) {
                            Edges.at(i)->setVisible(true);
                        }
                    }
                }
            }

        }else{
            font.setBold(false);
            font.setItalic(false);
            font.setUnderline(false);
            //font.setUnderline(false);
            if(edgeMode != EdgeMode_None){
                fillColor.setAlphaF(0.2);
                borderColor_selected.setAlphaF(0.2);
                borderColor_supervised.setAlphaF(0.2);
                borderColor_unselected.setAlpha(0.2);
                if(edgeMode == EdgeMode_Predecessor){
                    for (int i = 0; i < Edges.size(); ++i) {
                        Edges.at(i)->setVisible(false);
                    }
                }
                if(edgeMode == EdgeMode_Neighbours){
                    for (int i = 0; i < Edges.size(); ++i) {
                        if(Edges.at(i)->sourceNode->isSelected() != true && Edges.at(i)->destinationNode->isSelected() != true){
                            Edges.at(i)->setVisible(false);
                        }
                    }
                }
//                for (int i = 0; i < Edges.size(); ++i) {
//                    MyEdge* edge = Edges.at(i);
//                    if(edge->sourceNode->isSelected() != true && edge->destinationNode->isSelected() != true){
//                        edge->setVisible(false);
//                    }
//                }
            }else{
                fillColor.setAlphaF(1.0);
                borderColor_selected.setAlphaF(1.0);
                borderColor_supervised.setAlphaF(1.0);
                borderColor_unselected.setAlpha(1.0);
            }

        }

        if(isMarked == true){
            pen = QPen(QColor(0,0,0,255), 0);
            painter->setPen(pen);
            painter->drawEllipse(-radiusX, -radiusY, diameterX, diameterY);
        }

        if(isSupervised == true){
            font.setStrikeOut(true);
        }else{
            font.setStrikeOut(false);
        }
        font.setPointSize(diameterX/1.3);
        //font.setLetterSpacing(QFont::AbsoluteSpacing,-5);
        painter->setFont(font);
        pen = QPen(fillColor, fontSize);
        painter->setPen(pen);
        QRect boundingRect;
        painter->drawText(-radiusX, -radiusY, diameterX, diameterY,Qt::AlignCenter, text,&boundingRect);
        obj_boundingRect = boundingRect.adjusted(0, 0, -pen.width(), -pen.width());
        boundBoxCeilFactor = radiusY/3.1;
        obj_boundingRect.setY(obj_boundingRect.y()+boundBoxCeilFactor);
        obj_boundingRect.setHeight(obj_boundingRect.height()-boundBoxCeilFactor);
        //painter->drawRect(obj_boundingRect);
    }else if(drawOption == optionImage){
        QImage image(sampleImagePath);
        if(image.isNull()){
            return;
        }
        painter->drawImage(boundingRect(),image);
    }
}


//void MyNode::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
//{
//    QRadialGradient gradient(-radius*0.3, -radius*0.3, radius);
//    QPen pen;
//    QFont font = painter->font() ;
//    font.setPointSize(font.pointSize() / fontSizeScalingFactor);
//    font.setLetterSpacing(QFont::PercentageSpacing,0);
//    font.setStretch(QFont::UltraCondensed);
//    if(isSelected()){
//        font.setBold(true);
//        font.setItalic(true);
//    }else{
//        font.setBold(false);
//        font.setItalic(false);
//    }
//    painter->setFont(font);
//    if(isSupervised){
//        pen = QPen(fillColor, 0.8);
//        gradient.setCenter(radius*0.3, radius*0.3);
//        gradient.setFocalPoint(radius*0.3, radius*0.3);
//        gradient.setColorAt(0, fillColor.light(60));
//        gradient.setColorAt(1, fillColor.light(120));
//        painter->setBrush(gradient);
//        painter->setPen(pen);
//    }else{
//        pen = QPen(QColor(0,0,0,255), 0.8);
//        gradient.setCenter(radius*0.3, radius*0.3);
//        gradient.setFocalPoint(radius*0.3, radius*0.3);
//        painter->setPen(pen);
//    }
//    //text = "miau";
//    painter->drawText(boundingRect(), Qt::AlignCenter, text);
//}


//void MyNode::toolTipEvent(){
//    qDebug() << "miau";
//}

//void MyNode::hoverEnterEvent ( QGraphicsSceneHoverEvent * event ){
//    qDebug() << event->ty;
//    if(event->type() == QEvent::ToolTip){
//        qDebug() << "miau";
//    }
//}
