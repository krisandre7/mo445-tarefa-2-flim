#include "iftLabelMapGraphicsItem.h"
#include <QPainter>

LabelMapGraphicsItem::LabelMapGraphicsItem(const iftImage *init, QGraphicsItem *parent) :
  QGraphicsItem(parent),
  bboxf(QPointF(0,0), QPointF(init->xsize,init->ysize)),
  labelMap(iftCopyImage(init)),
  colorMap(iftMaximumValue(init))
{}

LabelMapGraphicsItem::LabelMapGraphicsItem(int xsize, int ysize, QGraphicsItem *parent) :
  QGraphicsItem(parent),
  bboxf(QPointF(0,0), QPointF(xsize,ysize)),
  labelMap(iftCreateImage(xsize, ysize, 1)),
  colorMap()
{}

LabelMapGraphicsItem::~LabelMapGraphicsItem()
{
  iftDestroyImage(&labelMap);
}

QRectF LabelMapGraphicsItem::boundingRect() const
{
  return bboxf;
}

void LabelMapGraphicsItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
  // Surpress unused variable warnings
  (void) option;
  (void) widget;

  int x1, x2, y1, y2;
  bboxf.toRect().getCoords(&x1, &y1, &x2, &y2);

  iftVoxel u = {0,0,0};
  for (u.y = y1; u.y < y2; ++(u.y)) {
    for (u.x = x1; u.x < x2; ++(u.x)) {
      iftVoxel v = {(u.x - x1), (u.y - y1), 0};
      int p = iftGetVoxelIndex(labelMap, v);
      if (labelMap->val[p] <= 0)
        continue;

      painter->setPen(colorMap[labelMap->val[p] - 1]);
      painter->drawPoint(u.x, u.y);
    }
  }
}

void LabelMapGraphicsItem::updateLabelMap(const iftImage *refImg)
{
  assert(refImg->n == labelMap->n);

  addMissingColors(iftMaximumValue(refImg));

  for (int i = 0; i < refImg->n; ++i)
    labelMap->val[i] = refImg->val[i];

  this->update();
}

void LabelMapGraphicsItem::setLabelColor(int label, QColor color, float opacity)
{
  assert(label > 0);
  addMissingColors(label);
  colorMap[label - 1] = color;
  colorMap[label - 1].setAlphaF(opacity);
}

void LabelMapGraphicsItem::addMissingColors(int newCount)
{
  int prevCount = colorMap.size();
  if (prevCount >= newCount)
    return;

  const float goldenRatioConjugate = 0.618033988749895f;
  colorMap.resize(newCount);
  for (int i = prevCount; i < newCount; ++i) {
    if (i == 0) {
      colorMap[i] = Qt::yellow;
    } else if (i == 1) {
      colorMap[i] = Qt::darkBlue;
    } else {
      float h = colorMap[i-1].hueF();
      h += goldenRatioConjugate;
      h = std::fmod(h, 1.0f);
      colorMap[i].setHsv(h, 0.8, 0.95); 
    }
    colorMap[i].setAlphaF(0.95);
  }
}

