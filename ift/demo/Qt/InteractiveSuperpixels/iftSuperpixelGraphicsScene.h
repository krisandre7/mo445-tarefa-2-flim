#ifndef _SUPERPIXEL_GRAPHICS_SCENE_H_
#define _SUPERPIXEL_GRAPHICS_SCENE_H_

#include <QGraphicsScene>
#include <QGraphicsItem>
#include <ift.h>
#include "iftRectContainerGraphicsItem.h"
#include "iftSuperpixelGraphicsItem.h"
#include "iftLabelMapGraphicsItem.h"

enum class SegVisualization {
  Hidden,
  LabelColors,
  Contrast,
  Borders
};

class iftSuperpixelGraphicsScene : public QGraphicsScene
{
  Q_OBJECT

  public:
    explicit iftSuperpixelGraphicsScene(const std::vector<iftSuperpixelIGraph *>& spIGraphs, const char *origPath, const iftImage *pxSeg = nullptr, QObject *parent = nullptr);
    virtual ~iftSuperpixelGraphicsScene();

  signals:
    void markerAdded(int level, int spLabel, int marker);

  public slots:
    void updateActiveLevel(int newActiveLevel);
    void updateSegmentation(const iftImage *segImg);
    void updateSeeds(const iftImage *seedImg);
    void updateSuperpixelOrdering(int scale, std::vector<int> &sortedSp);
    void setPxSegmentationVisualization(SegVisualization visType);
    void setPxSeedVisibility(bool isVisible);
    void setSpHierarchyVisibility(bool isVisible);

  protected:
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;

  private:
    // Helper functions
    iftImage * BuildBorderImage(const iftImage *labelMap);
    void BuildSuperpixelGraphicsItems(uint i);

    // Segmentation data
    // Change Rep to RectContainer if more GraphicsItems are added 
    SegVisualization segVis;
    LabelMapGraphicsItem * pxSegmentationRep;
    iftImage * pxSegLabelMap;
    iftImage * pxSegBorders;

    // Seed data
    // Change Rep to RectContainer if more GraphicsItems are added 
    LabelMapGraphicsItem * pxSeedRep;

    // Superpixel hierarchy data
    int activeLevel;
    iftRectContainerGraphicsItem * spHierarchyRep;
    std::vector<iftRectContainerGraphicsItem *> levelReps;
    std::vector<std::vector<SuperpixelGraphicsItem *>> superpixels;
    std::vector<const iftSuperpixelIGraph *> graphs;
    std::vector<iftImage *> borderImgs;

    // Superpixel highlighting from Active Learning helpers
    uint nHighlight; // Arbitrary for now, should be a setting
    std::vector<iftSet *> highlightedSp;

    // Mouse event helpers
    QPoint lastMousePos;
    int pressLabel;
    Qt::MouseButton pressButton;
};

#endif // _SUPERPIXEL_GRAPHICS_SCENE_H_

