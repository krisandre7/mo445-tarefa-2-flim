#ifndef _IFT_SEGMENTATION_THREAD_H_
#define _IFT_SEGMENTATION_THREAD_H_

// Pixel based segmentation only

#include <QThread>
#include <ift.h>

enum class SegMethod {
  Undefined,
  Watershed,
  GraphCut,
  DynamicSetObjectPolicy,
  DynamicSetRootPolicy,
  DynamicSetMinRootPolicy
};

class iftSegmentationThread : public QThread
{
  Q_OBJECT

  public:
    explicit iftSegmentationThread(SegMethod segMethod, iftImage *origImg, iftImage *markersImg, iftImage *gtImg = nullptr, QObject *parent = nullptr);
    virtual ~iftSegmentationThread();

  signals:
    void segResult(iftImage *res, double dice, float time);

  protected:
    void run();

  private:
    SegMethod method;
    iftImage *img;
    iftImage *markers;
    iftImage *gt;
};

iftLabeledSet * ConvertMarkersFromImageToLabeledSet(iftImage *markersImg, bool decrement = false);

#endif // _IFT_SEGMENTATION_THREAD_H_

