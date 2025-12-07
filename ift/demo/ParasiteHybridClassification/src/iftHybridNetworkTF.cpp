#include "iftHybridNetworkTF.h"

void iftVggLoadImg_TF(std::vector<float>&data, int counter, iftImage *img)
{
  assert(img != NULL);
  assert(iftIsColorImage(img));
  assert(img->xsize == PHN::width);
  assert(img->ysize == PHN::height);

  iftImage* R = iftImageRed(img);
  iftImage* G = iftImageGreen(img);
  iftImage* B = iftImageBlue(img);

  iftVggLoadImgFromBuffer_TF(data, counter, R->val, G->val, B->val, iftImageDepth(img));

  iftDestroyImage(&R);
  iftDestroyImage(&G);
  iftDestroyImage(&B);
}

void iftVggLoadImgFromPath_TF(std::vector<float>&data, int counter, const char *img_path)
{
  assert(img_path != NULL);

  iftImage* img = iftReadImageByExt(img_path);
  iftVggLoadImg_TF(data, counter, img);
  iftDestroyImage(&img);
}

void iftVggLoadImgFromBuffer_TF(std::vector<float>&data, int counter, int *R, int *G, int *B, int depth)
{

  float depthCorrection = (float) (1 << (depth - 8));

  // Should match kerasVggUtils.py:LoadDatasetWithVggPreProc
  // RGB->BGR, 8-bit depth conversion and subtract ImageNet mean (Initialization for Vgg16)
  int buf_idx = 0;
  int data_idx = counter * PHN::imgLen;
  for (int y = 0; y < PHN::height; ++y) {
    for (int x = 0; x < PHN::width; ++x) {
      data[data_idx++] = 
          ((float) B[buf_idx]) / depthCorrection - 103.939f;
      data[data_idx++] = 
          ((float) G[buf_idx]) / depthCorrection - 116.779f;
      data[data_idx++] = 
          ((float) R[buf_idx]) / depthCorrection - 123.68f;
      buf_idx++;
    }
  }
}
