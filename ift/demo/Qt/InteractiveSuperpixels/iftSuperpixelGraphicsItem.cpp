#include "iftSuperpixelGraphicsItem.h"
#include <QPainter>

SuperpixelGraphicsItem::SuperpixelGraphicsItem(const iftSuperpixelIGraph *spGraph, int spLabel, QRectF boundingBox, const iftImage *borderImage, QGraphicsItem *parent) :
  QGraphicsItem(parent),
  graph(spGraph),
  borderImg(borderImage),
  label(spLabel),
  bboxf(boundingBox),
  isHover(false),
  isHighlighted(false)
{}

SuperpixelGraphicsItem::~SuperpixelGraphicsItem()
{}

QRectF SuperpixelGraphicsItem::boundingRect() const
{
  return bboxf;
}

void SuperpixelGraphicsItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
  // Surpress unused variable warnings
  (void) option;
  (void) widget;

  int x1, x2, y1, y2;
  bboxf.toRect().getCoords(&x1, &y1, &x2, &y2);

  iftVoxel u = {0,0,0};
  for (u.y = y1; u.y < y2; ++(u.y)) {
    for (u.x = x1; u.x < x2; ++(u.x)) {
      int p = iftGetVoxelIndex(graph->refImg, u);
      if (graph->refImg->val[p] != label + 1)
        continue;

      if (borderImg && borderImg->val[p] > 0) {
        if (isHover) { 
          painter->setPen(Qt::green);
          painter->setOpacity(0.8);
        } else if (isHighlighted) {
          painter->setPen(Qt::darkGreen);
          painter->setOpacity(0.8);
        } else {
          painter->setPen(Qt::red);
          painter->setOpacity(0.4);
        }
        painter->drawPoint(u.x, u.y);
      }
    }
  }
}

void SuperpixelGraphicsItem::setHoverStatus(bool val)
{
  if (val != isHover) {
    isHover = val;
    this->update();
  }
}

void SuperpixelGraphicsItem::setHighlight(bool val)
{
  if (val != isHighlighted) {
    isHighlighted = val;
    this->update();
  }
}

