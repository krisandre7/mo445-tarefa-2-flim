#ifndef _IFT_HYBRID_NETWORK_TF_H_
#define _IFT_HYBRID_NETWORK_TF_H_

#include <vector>
#include <ift.h>

// Currently with hardcoded values from our problem for convenience
// PHN stands for Parasite Hybrid Network
namespace PHN {
    static int channels = 3;
    static int width = 200;
    static int height = 200;
    static int imgLen = channels * width * height;
}

void iftVggLoadImgFromPath_TF(std::vector<float>&data, int counter, const char *img_path);
void iftVggLoadImg_TF(std::vector<float>&data, int counter, iftImage *img);
void iftVggLoadImgFromBuffer_TF(std::vector<float>&data, int counter, int *R, int *G, int *B, int depth);

#endif // _IFT_HYBRID_NETWORK_TF_H_
