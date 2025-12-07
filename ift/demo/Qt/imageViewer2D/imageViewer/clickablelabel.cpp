#include "clickablelabel.h"


ClickableLabel::ClickableLabel( QWidget* parent)
    : QLabel(parent)
{
    setText("");
}

ClickableLabel::~ClickableLabel()
{
}

void ClickableLabel::mousePressEvent(QMouseEvent* ev)
{
    const QPoint mousePosition = ev->pos();
    emit mousePress(mousePosition);
}

void ClickableLabel::mouseMoveEvent(QMouseEvent* ev){
    const QPoint mousePosition = ev->pos();
    if(mousePosition.x() < this->width() && mousePosition.y()<this->height()){
        emit mouseHolding(mousePosition);
    }
}

void ClickableLabel::mouseReleaseEvent(QMouseEvent* ev){
    const QPoint mousePosition = ev->pos();
    emit mouseRelease(mousePosition);
}




