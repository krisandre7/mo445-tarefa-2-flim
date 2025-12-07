#ifndef _MAIN_WINDOW_H_
#define _MAIN_WINDOW_H_

#include <QMainWindow>
#include <ift.h>
#include "iftSuperpixelGraphicsScene.h"
#include "iftSegmentationThread.h"

class MainWindow : public QMainWindow
{
  Q_OBJECT

  public:
    explicit MainWindow(const char *origPath, char * const spLabelPaths[], int nLevels, const char *gtPath = nullptr);
    ~MainWindow();

  signals:
    // Changes to the RISF hierarchy granularity level
    void activeLevelChanged(int newActiveLevel);
    // Label map pxLabel updated
    void segmentationChanged(const iftImage *segImg);
    void seedsChanged(const iftImage *seedImg);
    void segVisualizationChanged(SegVisualization visType);
    void seedVisibilityChanged(bool isVisible);
    void spHierarchyVisibilityChanged(bool isVisible);

  public slots:
    void addMarker(int level, int spLabel, int marker);
    void updateSeg(iftImage *res, double dice, float time);
    void segThreadFinished();

  protected:
    void keyPressEvent(QKeyEvent *event);

  private:
    void LoadSuperpixelGraphs(char * const spLabelPaths[]);
    iftSuperpixelGraphicsScene * CreateScene(const char *origPath);
    void UpdateWindowTitle();
    void ComputeNewSegmentation(SegMethod seg);

    uint activeLevel;
    iftImage *origImg;
    iftImage *pxMarkers;
    iftImage *pxLabel;
    iftImage *gtImg;
    std::vector<iftSuperpixelIGraph *> spIGraphs;
    std::vector<iftImage *> spBorderDistance;
    SegMethod currentSeg;
    iftSegmentationThread *segThread;
    bool segHasUpToDateMarkers;
    
    // Visualization tracker
    SegVisualization pxSegVisualization;
    bool pxSeedIsVisible;
    bool spHierarchyIsVisible;

    // For now just rollback for a single addMarker call
    void prevMarkersUpdate(bool isRollback);
    bool rollbackAvailable;
    std::vector<int *> prevIGraphMarkers;
    iftImage *prevPxMarkers;
};

#endif // _MAIN_WINDOW_H_

