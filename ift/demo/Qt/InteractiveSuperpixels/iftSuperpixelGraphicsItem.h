#ifndef _SUPERPIXEL_GRAPHICS_ITEM_H_
#define _SUPERPIXEL_GRAPHICS_ITEM_H_

#include <QGraphicsItem>
#include <ift.h>

class SuperpixelGraphicsItem : public QGraphicsItem
{
  public:
    explicit SuperpixelGraphicsItem(const iftSuperpixelIGraph *spGraph, int spLabel, QRectF boundingBox, const iftImage *borderImage = nullptr, QGraphicsItem *parent = nullptr);
    ~SuperpixelGraphicsItem();

    QRectF boundingRect() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;

    void setHoverStatus(bool val);
    void setHighlight(bool val);
  private:
    const iftSuperpixelIGraph *graph;
    const iftImage *borderImg;
    const iftImage *segLabels;
    int label;
    QRectF bboxf;
    bool isHover;
    bool isHighlighted;
};

#endif
