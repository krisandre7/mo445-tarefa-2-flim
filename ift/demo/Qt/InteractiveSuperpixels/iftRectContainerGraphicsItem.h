#ifndef _IFT_RECT_CONTAINER_GRAPHICS_ITEM_H_
#define _IFT_RECT_CONTAINER_GRAPHICS_ITEM_H_

#include <QGraphicsItem>
#include <ift.h>

class iftRectContainerGraphicsItem : public QGraphicsItem
{
  public:
    explicit iftRectContainerGraphicsItem(int xSize, int ySize, QGraphicsItem *parent = NULL, QPointF offset = QPointF(0,0));
    virtual ~iftRectContainerGraphicsItem();

    QRectF boundingRect() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;

  private:
    QRectF bboxf;
};

#endif // _IFT_HIERARCHY_RECT_CONTAINER_ITEM_H_

