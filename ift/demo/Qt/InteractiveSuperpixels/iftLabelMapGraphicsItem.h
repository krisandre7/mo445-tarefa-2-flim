#ifndef _IFT_LABEL_MAP_GRAPHICS_ITEM_H_
#define _IFT_LABEL_MAP_GRAPHICS_ITEM_H_

#include <QGraphicsItem>
#include <QColor>
#include <ift.h>

class LabelMapGraphicsItem : public QGraphicsItem
{
  public:
    explicit LabelMapGraphicsItem(const iftImage *initLabelMap, QGraphicsItem *parent = nullptr);
    explicit LabelMapGraphicsItem(int xsize, int ysize, QGraphicsItem *parent = nullptr);
    ~LabelMapGraphicsItem();

    QRectF boundingRect() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;

    // First two new unset label colors are set to yellow and blue.
    //   Color choice is based on fixed hue shift afterwards.
    // Actual labels start from 1
    void updateLabelMap(const iftImage *refImg);
    void setLabelColor(int label, QColor color, float opacity = 0.5);
  private:
    void addMissingColors(int newCount);
    QRectF bboxf;
    iftImage *labelMap;
    std::vector<QColor> colorMap;
};

#endif // _IFT_LABEL_MAP_GRAPHICS_ITEM_H_
