#include "iftRectContainerGraphicsItem.h"

iftRectContainerGraphicsItem::iftRectContainerGraphicsItem(int xSize, int ySize, QGraphicsItem *parent, QPointF offset) :
  QGraphicsItem(parent),
  bboxf(offset, QPointF(xSize, ySize))
{}

iftRectContainerGraphicsItem::~iftRectContainerGraphicsItem()
{}

QRectF iftRectContainerGraphicsItem::boundingRect() const
{
  return bboxf;
}

void iftRectContainerGraphicsItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
  // Surpress unused variable warnings
  (void) painter;
  (void) option;
  (void) widget;
}

