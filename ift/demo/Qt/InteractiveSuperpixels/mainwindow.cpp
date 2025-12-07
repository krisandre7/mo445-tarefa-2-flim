#include  <QtWidgets>
#include "mainwindow.h"

// Private in iftRISF.c
extern "C" {
  iftMatrix * iftComputeSuperpixelFeaturesByColorSpaceMean(const iftImage *superpixelLabelMap, const iftImage *img, iftColorSpace colorSpace);  
}

MainWindow::MainWindow(const char *origPath, char * const spLabelPaths[], int nLevels, const char *gtPath) :
  activeLevel(0),
  origImg(iftReadImageByExt(origPath)),
  pxMarkers(nullptr),
  pxLabel(nullptr),
  gtImg(nullptr),
  spIGraphs(nLevels),
  spBorderDistance(nLevels),
  currentSeg(SegMethod::Undefined),
  segThread(nullptr),
  segHasUpToDateMarkers(true),
  pxSegVisualization(SegVisualization::Hidden),
  pxSeedIsVisible(false),
  spHierarchyIsVisible(true),
  rollbackAvailable(false),
  prevIGraphMarkers(nLevels),
  prevPxMarkers(nullptr)
{
  if (gtPath)
    gtImg = iftReadImageByExt(gtPath);
  pxMarkers = iftCreateImage(origImg->xsize, origImg->ysize, origImg->zsize);
  // Temporary until a better history/rollback is implemented
  prevPxMarkers = iftCreateImage(origImg->xsize, origImg->ysize, origImg->zsize);
  pxLabel = iftCreateImage(origImg->xsize, origImg->ysize, origImg->zsize);

  this->LoadSuperpixelGraphs(spLabelPaths);
  
  iftSuperpixelGraphicsScene * scene = this->CreateScene(origPath);
  emit activeLevelChanged(activeLevel);

  QGraphicsView *view = new QGraphicsView(this);
  view->setScene(scene);
  view->setMouseTracking(true);

  setCentralWidget(view);

  this->UpdateWindowTitle();
  this->grabKeyboard();
}

MainWindow::~MainWindow()
{
  iftDestroyImage(&origImg);
  iftDestroyImage(&pxMarkers);
  iftDestroyImage(&pxLabel);
  iftDestroyImage(&gtImg);
  for (uint i = 0; i < spIGraphs.size(); ++i)
    iftDestroySuperpixelIGraph(&(spIGraphs[i]));
}

void MainWindow::addMarker(int level, int spLabel, int marker)
{
  assert(level >= 0 && (uint) level < spIGraphs.size());

  // Temporary until a better history/rollback is implemented
  prevMarkersUpdate(false);

  // If covering all pixels becomes a bottleneck,
  //   it is possible to pre-calculate the
  //   superpixels' children across levels
  for (int p = 0; p < origImg->n; ++p) {
    if (spIGraphs[level]->refImg->val[p] != spLabel + 1)
      continue;

    // Implicit erosion
    if (spBorderDistance[level]->val[p] <= 2*2)
      continue;

    for (uint l = level; l < spIGraphs.size(); ++l) {
      int sp = spIGraphs[l]->refImg->val[p] - 1;
      spIGraphs[l]->ann->marker[sp] = marker;
    }
    pxMarkers->val[p] = marker;
  }

  emit seedsChanged(pxMarkers);
  pxSeedIsVisible = true;
  emit seedVisibilityChanged(pxSeedIsVisible);

  if (currentSeg != SegMethod::Undefined) {
    segHasUpToDateMarkers = false;
    UpdateWindowTitle();
  }
}

void MainWindow::updateSeg(iftImage *res, double dice, float time)
{
  // TODO change to swap & destroy
  for (int p = 0; p < res->n; ++p)
    pxLabel->val[p] = res->val[p];

  printf("Segmentation performed in %fms with dice = %lf\n", time, dice);
  emit segmentationChanged(pxLabel);
  if (pxSegVisualization == SegVisualization::Hidden) {
    pxSegVisualization = SegVisualization::Borders;
    emit segVisualizationChanged(pxSegVisualization); 
  }

  iftDestroyImage(&res);
}

void MainWindow::segThreadFinished()
{
  delete segThread;
  segThread = nullptr;
  UpdateWindowTitle();
}

void MainWindow::keyPressEvent(QKeyEvent *event)
{
  QMainWindow::keyPressEvent(event);
  uint prevActiveLevel = activeLevel;

  // Change hierarchy level
  if (event->key() == Qt::Key_Up) {
    if (activeLevel < spIGraphs.size() - 1)
      activeLevel += 1;

    // Force visualization of result
    spHierarchyIsVisible = true;
    emit spHierarchyVisibilityChanged(spHierarchyIsVisible); 
  }

  if (event->key() == Qt::Key_Down) {
    if (activeLevel > 0)
      activeLevel -= 1;

    // Force visualization of result
    spHierarchyIsVisible = true;
    emit spHierarchyVisibilityChanged(spHierarchyIsVisible); 
  }

  // Rollback - Temporary barebones version for markers
  if (event->key() == Qt::Key_Z) {
    if (rollbackAvailable) {
      prevMarkersUpdate(true);

      if (currentSeg != SegMethod::Undefined) {
        segHasUpToDateMarkers = false;
        UpdateWindowTitle();
      }
    } else {
      printf("Rollback ignored. Already at oldest available markers.\n");
    }

    // Force visualization of result
    emit seedsChanged(pxMarkers); 
    pxSeedIsVisible = true;
    emit seedVisibilityChanged(pxSeedIsVisible); 
  }

  if (activeLevel != prevActiveLevel) {
    UpdateWindowTitle();
    emit activeLevelChanged(activeLevel);
  }

  // Change active segmentation
  if (event->key() >= Qt::Key_0 && event->key() <= Qt::Key_9) {
    SegMethod seg = SegMethod::Undefined;
    switch (event->key()) {
      case Qt::Key_1:
        seg = SegMethod::Watershed;
        break;
      case Qt::Key_2:
        seg = SegMethod::GraphCut;
        break;
      case Qt::Key_3:
        seg = SegMethod::DynamicSetObjectPolicy;
        break;
      case Qt::Key_4:
        seg = SegMethod::DynamicSetRootPolicy;
        break;
      case Qt::Key_5:
        seg = SegMethod::DynamicSetMinRootPolicy;
        printf("Warning: DynamicSetMinRoot is slow due to complexity O(nSeeds)\n");
        break;
      default:
        seg = SegMethod::Undefined;
    }
    ComputeNewSegmentation(seg);
  }

  // Toggle Visibility
  if (event->key() == Qt::Key_Q) {
    // Cycle through visualization types
    switch (pxSegVisualization) {
      case SegVisualization::Hidden:
        pxSegVisualization = SegVisualization::Borders;
        break;
      case SegVisualization::Borders:
        pxSegVisualization = SegVisualization::Contrast;
        break;
      case SegVisualization::Contrast:
        pxSegVisualization = SegVisualization::LabelColors;
        break;
      case SegVisualization::LabelColors:
        pxSegVisualization = SegVisualization::Hidden;
        break;
    }
    emit segVisualizationChanged(pxSegVisualization); 
    UpdateWindowTitle();
  }

  if (event->key() == Qt::Key_E) {
    pxSeedIsVisible = !pxSeedIsVisible;
    emit seedVisibilityChanged(pxSeedIsVisible);
  }

  if (event->key() == Qt::Key_W) {
    spHierarchyIsVisible = !spHierarchyIsVisible;
    emit spHierarchyVisibilityChanged(spHierarchyIsVisible); 
  }

  // Compute ordering by active learning
  if (event->key() == Qt::Key_A) {
    // Active Learning by X
    // Analogous to segmentation methods but with active learning
  }
}

void MainWindow::LoadSuperpixelGraphs(char * const spLabelPaths[])
{
  bool is3D = iftIs3DImage(this->origImg); 
  iftAdjRel *A = is3D ? iftSpheric(1.0) : iftCircular(1.0);

  for (uint i = 0; i < spIGraphs.size(); ++i) {
    iftImage *refImg = iftReadImageByExt(spLabelPaths[i]);
    spIGraphs[i] = iftInitSuperpixelIGraph(refImg);
    spBorderDistance[i] = iftBorderDistTrans(spIGraphs[i]->refImg, A);
    // Temporary until a better history/rollback is implemented
    prevIGraphMarkers[i] = 
      (int *) malloc(spIGraphs[i]->ann->n * sizeof(int));
    iftCopyIntArray(prevIGraphMarkers[i],
        spIGraphs[i]->ann->marker, spIGraphs[i]->ann->n);

    iftSetSuperpixelIGraphExplicitRAG(spIGraphs[i], A);

    // For now we set color mean as feature by default
    // And fixed alpha/beta (which might not even be used)
    bool isColorImage = iftIsColorImage(this->origImg);
    float alpha = isColorImage ? 0.4 : 40.96;
    float beta = 8;
    iftMatrix *pFeats = NULL;

    if (isColorImage)
      pFeats = iftComputeSuperpixelFeaturesByColorSpaceMean(refImg, origImg, LABNorm_CSPACE);  
    else
      pFeats = iftComputeSuperpixelFeaturesByColorSpaceMean(refImg, origImg, GRAYNorm_CSPACE);  

    iftSetSuperpixelIGraphFeatures(spIGraphs[i], pFeats, iftDistance1, alpha, beta);

    iftDestroyImage(&refImg);
    iftDestroyMatrix(&pFeats);
  }

  iftDestroyAdjRel(&A);
}

iftSuperpixelGraphicsScene * MainWindow::CreateScene(const char *origPath)
{
  assert(spIGraphs.size() > 0);

  // GraphicsItems are created within the ctor
  iftSuperpixelGraphicsScene *scene = 
    new iftSuperpixelGraphicsScene(spIGraphs, origPath, pxLabel, this);
  
  // Hierarchy level change
  QObject::connect(
      this, SIGNAL(activeLevelChanged(int)),
      scene, SLOT(updateActiveLevel(int)));
  // Segmentation label map update
  QObject::connect(
      this, SIGNAL(segmentationChanged(const iftImage *)),
      scene, SLOT(updateSegmentation(const iftImage *)));
  // Segmentation visualization update
  QObject::connect(
      this, SIGNAL(segVisualizationChanged(SegVisualization)),
      scene, SLOT(setPxSegmentationVisualization(SegVisualization)));
  // Seeds label map update
  QObject::connect(
      this, SIGNAL(seedsChanged(const iftImage *)),
      scene, SLOT(updateSeeds(const iftImage *)));
  // Seeds visualization update
  QObject::connect(
      this, SIGNAL(seedVisibilityChanged(bool)),
      scene, SLOT(setPxSeedVisibility(bool)));
  // Sp Hierarchy visualization update
  QObject::connect(
      this, SIGNAL(spHierarchyVisibilityChanged(bool)),
      scene, SLOT(setSpHierarchyVisibility(bool)));
  // User added marker by clicking superpixel
  QObject::connect(
      scene, SIGNAL(markerAdded(int,int,int)),
      this, SLOT(addMarker(int,int,int)));

  return scene;
}

void MainWindow::UpdateWindowTitle()
{
  char buf[255];
  int len = 0;

  // Application name
  len += snprintf(buf + len, 255 - len, "InteractiveSuperpixels");

  // Hierarchy level
  len += snprintf(buf + len, 255 - len,
      " - Level %d/%lu", activeLevel+1, spIGraphs.size());  

  // Segmentation
  if (segThread != nullptr) {
    len += snprintf(buf + len, 255 - len,
        " - Running segmentation");
  } else {
    switch (currentSeg) {
      case SegMethod::Undefined:
        len += snprintf(buf + len, 255 - len, 
            " - No segmentation");
        break;
      case SegMethod::Watershed:
        len += snprintf(buf + len, 255 - len,
            " - Watershed");
        break;
      case SegMethod::GraphCut:
        len += snprintf(buf + len, 255 - len,
            " - GraphCut");
        break;
      case SegMethod::DynamicSetObjectPolicy:
        len += snprintf(buf + len, 255 - len,
            " - DynamicIFT(Obj)");
        break;
      case SegMethod::DynamicSetRootPolicy:
        len += snprintf(buf + len, 255 - len,
            " - DynamicIFT(Root)");
        break;
      case SegMethod::DynamicSetMinRootPolicy:
        len += snprintf(buf + len, 255 - len,
            " - DynamicIFT(MinRoot)");
        break;
    }

    if (currentSeg != SegMethod::Undefined) {
      switch (pxSegVisualization) {
        case SegVisualization::Hidden:
          len += snprintf(buf + len, 255 - len,
              " (Hidden)");
          break;
        case SegVisualization::LabelColors:
          len += snprintf(buf + len, 255 - len,
              " (Colors)");
          break;
        case SegVisualization::Contrast:
          len += snprintf(buf + len, 255 - len,
              " (Contrast)");
          break;
        case SegVisualization::Borders:
          len += snprintf(buf + len, 255 - len,
              " (Borders)");
          break;
      }
    }
  }

  // Pending updates
  if (!segHasUpToDateMarkers)
    len += snprintf(buf + len, 255 - len, "*");

  setWindowTitle(tr(buf));
}

void MainWindow::ComputeNewSegmentation(SegMethod seg)
{
  // Ignore when a thread is already running
  if (segThread != nullptr) {
    printf("Segmentation already running!\n");
    return;
  }

  segHasUpToDateMarkers = true;
  currentSeg = seg;
  if (seg == SegMethod::Undefined) {
    // Clear current segmentation if none is defined
    for (int p = 0; p < pxLabel->n; ++p)
      pxLabel->val[p] = 0;
    emit segmentationChanged(pxLabel);
    UpdateWindowTitle();
  } else {
    // Otherwise leave a thread working on the segmentation
    segThread =
      new iftSegmentationThread(seg, origImg, pxMarkers, gtImg);
    QObject::connect(
        segThread, SIGNAL(segResult(iftImage*,double,float)),
        this, SLOT(updateSeg(iftImage *, double, float)),
        Qt::QueuedConnection);
    QObject::connect(
        segThread, SIGNAL(finished()),
        this, SLOT(segThreadFinished()),
        Qt::QueuedConnection);
    segThread->start();
    currentSeg = seg;
    UpdateWindowTitle();
  }
}

void MainWindow::prevMarkersUpdate(bool isRollback)
{
  if (isRollback) {
    rollbackAvailable = false;
    for (uint l = 0; l < spIGraphs.size(); ++l)
      for (int sp = 0; sp < spIGraphs[l]->ann->n; ++sp)
          spIGraphs[l]->ann->marker[sp] = prevIGraphMarkers[l][sp];
    for (int p = 0; p < origImg->n; ++p)
      pxMarkers->val[p] = prevPxMarkers->val[p];
  } else {
    for (uint l = 0; l < spIGraphs.size(); ++l)
      for (int sp = 0; sp < spIGraphs[l]->ann->n; ++sp)
        prevIGraphMarkers[l][sp] = spIGraphs[l]->ann->marker[sp];
    for (int p = 0; p < origImg->n; ++p)
      prevPxMarkers->val[p] = pxMarkers->val[p];
    rollbackAvailable = true;
  }
}

