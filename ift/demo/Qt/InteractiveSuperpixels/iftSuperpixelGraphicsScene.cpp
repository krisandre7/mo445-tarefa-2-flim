#include "iftSuperpixelGraphicsScene.h"
#include <QGraphicsSceneMouseEvent>

iftSuperpixelGraphicsScene::iftSuperpixelGraphicsScene(const std::vector<iftSuperpixelIGraph *>& spIGraphs, const char *origPath, const iftImage *pxSeg, QObject *parent) :
  QGraphicsScene(parent),
  segVis(SegVisualization::Hidden),
  pxSegmentationRep(nullptr),
  pxSegLabelMap(nullptr),
  pxSegBorders(nullptr),
  pxSeedRep(nullptr),
  activeLevel(0),
  spHierarchyRep(nullptr),
  levelReps(spIGraphs.size()),
  superpixels(spIGraphs.size()),
  graphs(spIGraphs.size()),
  borderImgs(spIGraphs.size()),
  nHighlight(5), // Arbitrary
  highlightedSp(spIGraphs.size()),
  lastMousePos(-1,-1),
  pressLabel(-1),
  pressButton(Qt::NoButton)
{
  assert(spIGraphs.size() > 0);

  // Build segmentation items
  pxSegmentationRep = new LabelMapGraphicsItem(pxSeg->xsize, pxSeg->ysize);
  pxSegLabelMap = iftCreateImage(pxSeg->xsize, pxSeg->ysize, 1);
  pxSegBorders = iftCreateImage(pxSeg->xsize, pxSeg->ysize, 1);

  // Build seed items
  pxSeedRep = new LabelMapGraphicsItem(pxSeg->xsize, pxSeg->ysize);
  pxSeedRep->setLabelColor(1, Qt::yellow, 0.6f);
  pxSeedRep->setLabelColor(2, Qt::darkBlue, 0.6f);

  // Build superpixel items
  spHierarchyRep = new iftRectContainerGraphicsItem(spIGraphs[0]->refImg->xsize, spIGraphs[0]->refImg->ysize);
  for (uint i = 0; i < spIGraphs.size(); ++i) {
    graphs[i] = spIGraphs[i];
    borderImgs[i] = this->BuildBorderImage(graphs[i]->refImg);
    this->BuildSuperpixelGraphicsItems(i);
    highlightedSp[i] = nullptr;
    levelReps[i]->setVisible(false);
  }
  levelReps[activeLevel]->setVisible(true);

  // Original image acts as background and never changes
  this->addPixmap(QPixmap(origPath));

  // Set initial visualization status
  pxSegmentationRep->setVisible(false);
  pxSeedRep->setVisible(true);
  spHierarchyRep->setVisible(true);

  // Add components to scene, order impacts simulataneous visualization
  this->addItem(pxSeedRep);
  this->addItem(spHierarchyRep);
  this->addItem(pxSegmentationRep);
}

iftSuperpixelGraphicsScene::~iftSuperpixelGraphicsScene()
{
  for (uint i = 0; i < borderImgs.size(); ++i)
    iftDestroyImage(&(borderImgs[i]));
}

void iftSuperpixelGraphicsScene::updateActiveLevel(int newActiveLevel)
{
  // Currently runs even if sp hierarchy is not visible
  if (activeLevel != newActiveLevel) {
    levelReps[activeLevel]->setVisible(false);
    levelReps[newActiveLevel]->setVisible(true);

    // Update hover status
    QPoint &p = lastMousePos;
    iftVoxel u = {p.x(), p.y(), 0};
    if (iftValidVoxel(graphs[activeLevel]->refImg, u)) {
      int hoverLabel = iftImgVal(graphs[activeLevel]->refImg,
            p.x(), p.y(), 0) - 1;
      int newHoverLabel = iftImgVal(graphs[newActiveLevel]->refImg,
          p.x(), p.y(), 0) - 1;
      superpixels[activeLevel][hoverLabel]->setHoverStatus(false);
      superpixels[newActiveLevel][newHoverLabel]->setHoverStatus(true);
    }

    // Reset click data
    pressButton = Qt::NoButton;

    activeLevel = newActiveLevel;
  }
}

void iftSuperpixelGraphicsScene::updateSegmentation(const iftImage *segImg)
{
  assert(segImg != nullptr);

  iftDestroyImage(&pxSegLabelMap);
  pxSegLabelMap = iftCopyImage(segImg);

  iftDestroyImage(&pxSegBorders);
  pxSegBorders = BuildBorderImage(pxSegLabelMap);

  setPxSegmentationVisualization(segVis);
}

void iftSuperpixelGraphicsScene::updateSeeds(const iftImage *seedImg)
{
  assert(seedImg != nullptr);
  pxSeedRep->updateLabelMap(seedImg);
}

void iftSuperpixelGraphicsScene::updateSuperpixelOrdering(int scale, std::vector<int> &sortedSp)
{
  // Remove previous highlights
  while (highlightedSp[scale] != nullptr) {
    int sp = iftRemoveSet(&(highlightedSp[scale]));
    superpixels[scale][sp]->setHighlight(false);
  }

  if (sortedSp.empty())
    return;

  // Add new ones
  for (uint i = 0; i < nHighlight && i < sortedSp.size(); ++i) {
    int sp = sortedSp[i];
    iftInsertSet(&(highlightedSp[scale]), sp);
    superpixels[scale][sp]->setHighlight(true);
  }
}

void iftSuperpixelGraphicsScene::setPxSegmentationVisualization(SegVisualization visType)
{
  // Store new visualization type
  segVis = visType;

  // Update representative label map graphics item according to type

  if (visType == SegVisualization::Hidden) {
    pxSegmentationRep->setVisible(false);
    return;
  }

  switch (visType) {
    case SegVisualization::LabelColors:
      pxSegmentationRep->updateLabelMap(pxSegLabelMap);
      pxSegmentationRep->setLabelColor(1, Qt::yellow, 0.4f);
      pxSegmentationRep->setLabelColor(2, Qt::blue, 0.4f);
      break;
    case SegVisualization::Contrast:
      pxSegmentationRep->updateLabelMap(pxSegLabelMap);
      pxSegmentationRep->setLabelColor(1, QColor(255, 0, 128), 1.0f);
      pxSegmentationRep->setLabelColor(2, Qt::blue, 0.0f);
      break;
    default:
    case SegVisualization::Borders:
      pxSegmentationRep->updateLabelMap(pxSegBorders);
      pxSegmentationRep->setLabelColor(1, QColor(255,255,255), 1.0f);
      break;
  }

  pxSegmentationRep->setVisible(true);
}

void iftSuperpixelGraphicsScene::setPxSeedVisibility(bool isVisible)
{
  pxSeedRep->setVisible(isVisible);
}

void iftSuperpixelGraphicsScene::setSpHierarchyVisibility(bool isVisible)
{
  spHierarchyRep->setVisible(isVisible);
}

void iftSuperpixelGraphicsScene::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
  iftImage *refImg = graphs[activeLevel]->refImg; // Shorthand
  QPoint p = event->scenePos().toPoint();
  QPoint &q = lastMousePos; // Shorthand
  iftVoxel u = {p.x(), p.y(), 0};
  iftVoxel v = {q.x(), q.y(), 0};
  int newLabel = !(iftValidVoxel(refImg, u)) ? -1 :
    iftImgVal(refImg, p.x(), p.y(), 0) - 1;
  int lastLabel = !(iftValidVoxel(refImg, v)) ? -1 :
    iftImgVal(refImg, q.x(), q.y(), 0) - 1;

  if (newLabel != lastLabel) {
    if (lastLabel >= 0)
      superpixels[activeLevel][lastLabel]->setHoverStatus(false);
    if (newLabel >= 0)
      superpixels[activeLevel][newLabel]->setHoverStatus(true);
  }

  lastMousePos = p;
}

void iftSuperpixelGraphicsScene::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
  // Ignore events when there is no active level
  if (activeLevel < 0)
    return;

  if (event->button() != Qt::LeftButton && event->button() != Qt::RightButton) {
    event->ignore();
    return;
  }

  QPoint p = event->scenePos().toPoint();
  iftImage *refImg = graphs[activeLevel]->refImg; // Shorthand
  pressLabel = iftImgVal(refImg, p.x(), p.y(), 0) - 1;
  pressButton = event->button();
}

void iftSuperpixelGraphicsScene::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
  // Ignore events when there is no active level
  if (activeLevel < 0)
    return;

  // Not the currently tracked mouse button
  if (pressButton != event->button())
    return;

  QPoint p = event->scenePos().toPoint();
  iftImage *refImg = graphs[activeLevel]->refImg; // Shorthand
  int releaseLabel = iftImgVal(refImg, p.x(), p.y(), 0) - 1;

  // Mouse moved outside superpixel after being pressed
  if (pressLabel != releaseLabel) {
    pressLabel = -1;
    pressButton = Qt::NoButton;
    return;
  }

  // It is a proper click, just add marker
  int marker = (pressButton == Qt::LeftButton) ? 2 : 1;
  emit markerAdded(activeLevel, pressLabel, marker);

  superpixels[activeLevel][pressLabel]->update();
}

iftImage * iftSuperpixelGraphicsScene::BuildBorderImage(const iftImage *labelMap)
{
  iftAdjRel *A = iftCircular(sqrt(2.0));
  iftImage *res = iftCreateImage(labelMap->xsize, labelMap->ysize, labelMap->zsize);

  for (int p = 0; p < labelMap->n; ++p) {
    iftVoxel u = iftGetVoxelCoord(labelMap, p);

    // Border is defined by having adjacent voxel with different label
    int isBorder = 0;
    for (int i = 1; i < A->n; ++i) {
      iftVoxel v = iftGetAdjacentVoxel(A, u, i);
      if (!iftValidVoxel(labelMap, v))
        continue;

      int q = iftGetVoxelIndex(labelMap, v);
      if (labelMap->val[p] != labelMap->val[q]) {
        isBorder = 1;
        break;
      }
    }

    res->val[p] = isBorder;
  }

  iftDestroyAdjRel(&A);
  return res;
}

void iftSuperpixelGraphicsScene::BuildSuperpixelGraphicsItems(uint i)
{
  iftImage *spImg = graphs[i]->refImg; // Shorthand

  // Dummy parent to allow operations over all
  //   superpixels of a given level
  levelReps[i] = new iftRectContainerGraphicsItem(spImg->xsize, spImg->ysize, spHierarchyRep);

  uint nSp = iftMaximumValue(spImg);
  std::vector<int> spX1(nSp);
  std::vector<int> spX2(nSp);
  std::vector<int> spY1(nSp);
  std::vector<int> spY2(nSp);

  for (uint j = 0; j < nSp; ++j) {
    spX1[j] = spImg->xsize;
    spY1[j] = spImg->ysize;
    spX2[j] = spY2[j] = 0;
  }

  for (int y = 0; y < spImg->ysize; ++y) {
    for (int x = 0; x < spImg->xsize; ++x) {
      int label = iftImgVal(spImg, x, y, 0) - 1;
      spX1[label] = iftMin(iftMax(spX1[label]-1, 0), x);
      spX2[label] = iftMax(iftMin(spX2[label]+1, spImg->xsize), x);
      spY1[label] = iftMin(iftMax(spY1[label]-1, 0), y);
      spY2[label] = iftMax(iftMin(spY2[label]+1, spImg->ysize), y);
    }
  }

  for (uint j = 0; j < nSp; ++j) {
    superpixels[i].push_back(new SuperpixelGraphicsItem(graphs[i], j,
          QRectF(QPointF(spX1[j], spY1[j]), QPointF(spX2[j],spY2[j])),
          borderImgs[i], levelReps[i]));
  }
}

