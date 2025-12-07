#include "iftSegmentationThread.h"

iftSegmentationThread::iftSegmentationThread(SegMethod segMethod, iftImage *origImg, iftImage *markersImg, iftImage *gtImg, QObject *parent) :
  QThread(parent),
  method(segMethod),
  img(iftCopyImage(origImg)),
  markers(iftCopyImage(markersImg)),
  gt(iftCopyImage(gtImg))
{}

iftSegmentationThread::~iftSegmentationThread()
{
  iftDestroyImage(&img);
  iftDestroyImage(&markers);
  iftDestroyImage(&gt);
}

void iftSegmentationThread::run()
{
  iftMImage *mimg = iftImageToMImage(img, LABNorm_CSPACE);
  iftLabeledSet *seeds = ConvertMarkersFromImageToLabeledSet(markers, true);
  iftAdjRel *A = iftCircular(1.0);
  iftImage *res = nullptr;
  double dice = 0.0;
  float time = 0.0f;
  timer *tic = NULL;
  timer *toc = NULL;

  // Actual segmentation
  tic = iftTic();
  switch (method) {
    default:
    case SegMethod::Watershed:
      res = iftWaterCut(mimg, A, seeds, NULL);
      break;
    case SegMethod::GraphCut:
      res = iftGraphCutFromMImage(mimg, seeds, 100);
      break;
    case SegMethod::DynamicSetObjectPolicy:
      res = iftDynamicSetObjectPolicy(mimg, A, seeds, false);
      break;
    case SegMethod::DynamicSetRootPolicy:
      res = iftDynTreeRoot(mimg, A, seeds, 1, 0, NULL, 0);
      break;
    case SegMethod::DynamicSetMinRootPolicy:
      res = iftDynTreeClosestRoot(mimg, A, seeds, 1, 0, NULL, 0);
      break;
  }
  toc = iftToc();

  // Metrics
  if (gt != NULL) {
    int gtMin = iftMinimumValue(gt);
    if (gtMin <= 0) {
      for (int i = 0; i < gt->n; ++i) {
        gt->val[i] -= gtMin;
        if (gt->val[i] > 1)
          gt->val[i] = 1;
      }
    }
    dice = iftDiceSimilarity(res, gt);
  }
  time = iftCompTime(tic, toc);

  // Send result & cleanup
  emit segResult(res, dice, time);

  iftDestroyLabeledSet(&seeds);
  iftDestroyMImage(&mimg);
  iftDestroyAdjRel(&A);
}

iftLabeledSet * ConvertMarkersFromImageToLabeledSet(iftImage *markersImg, bool decrement)
{
  iftLabeledSet *seeds = nullptr;

  for (int i = 0; i < markersImg->n; ++i) {
    int label = markersImg->val[i];
    if (label <= 0)
      continue;

    iftInsertLabeledSet(&seeds, i, label  - decrement);
  }

  return seeds;
}

