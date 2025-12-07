#include "ift.h"

int main(int argc, char *argv[]) 
{
  iftImage *img, *label, *filt_label;

  if (argc != 5)
    iftExit("iftFilterLabelImage <input-orig.*> <input-label.*> <input-area> <output-label.*>","iftFilterLabelImage");
  
  img   = iftReadImageByExt(argv[1]);
  label = iftReadImageByExt(argv[2]);
  iftImage *aux = label;
  label = iftSmoothRegionsByDiffusion(aux,img,0.5,2);
  iftDestroyImage(&aux);
  filt_label = iftSelectAndPropagateRegionsAboveAreaByColor(img,label,atoi(argv[3]));
  iftDestroyImage(&label);
  iftDestroyImage(&img);

  iftWriteImageByExt(filt_label,argv[4]);

  return(0);
}
