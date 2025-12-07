#include "ift.h"


void usage();

void iftWriteOverlay
(iftImage* orig, iftImage *label, const char *filename);

iftImage *iftSupressLargestSaliencyRegion(iftImage *label_img, iftImage *saliency_img);
iftImage *iftSupressLargestSaliencyGlobal(iftImage *saliency_img);
iftImage *iftSupressAdaptMeanSaliencyGlobal(iftImage *saliency_img);
//-------------------------------------------------------------

int main(int argc, const char *argv[]) 
{
  int num_seeds;
  float obj_perc;
  iftImage *objsm, *seed_img, *mask, *supr_objsm;

  if(argc < 5 || argc > 6) usage();

  objsm = iftReadImageByExt(argv[1]);
  num_seeds = atoi(argv[2]);
  obj_perc = atof(argv[3]);

  if(argc == 6) mask = iftReadImageByExt(argv[5]);
  else mask = NULL;

  // supr_objsm = iftSupressLargestSaliencyRegion(mask, objsm);
  // supr_objsm = iftSupressLargestSaliencyGlobal(objsm);
  supr_objsm = iftSupressAdaptMeanSaliencyGlobal(objsm);
  iftWriteImageByExt(supr_objsm,"supr.pgm");
  seed_img = iftSamplingByOSMOX(objsm, supr_objsm, num_seeds, obj_perc);

  printf("NUM_SEEDS: %d\n", iftNumberOfElements(seed_img));

  iftWriteOverlay(objsm, seed_img, argv[4]);

  iftDestroyImage(&objsm);
  iftDestroyImage(&seed_img);
  if(mask != NULL) iftDestroyImage(&mask);

  return (0);
}

//-------------------------------------------------------------

void usage()
{
  printf("Usage: iftOSMOXSampl [1] [2] [3] [4] {5}\n");
  printf("---------------------------------------------------------\n");
  printf("[1] - Gray-level object saliency map\n");
  printf("[2] - Number of seeds\n");
  printf("[3] - Percentage of seeds in the object\n");
  printf("[4] - Output image\n\n");
  printf("{5} - OPTIONAL: Binary mask\n");

  iftError("Too many/few args!", "main");
}

iftImage *iftSupressLargestSaliencyRegion(iftImage *label_img, iftImage *saliency_img)
{
  int max_label, largest_size, largest_label, largest_saliency;
  int *region_area, *region_saliency;
  iftImage *rlabel,*nlabel;
  iftAdjRel *A;

  if (iftIs3DImage(label_img)) A = iftSpheric(sqrtf(1.0));
  else A = iftCircular(sqrtf(1.0));

  rlabel = iftRelabelRegions(label_img,A);

  max_label = iftMaximumValue(rlabel);
  region_area = iftAllocIntArray(max_label+1);
  region_saliency = iftAllocIntArray(max_label+1);

  for(int p = 0; p < saliency_img->n; p++) {
    region_area[rlabel->val[p]]++;
    region_saliency[rlabel->val[p]] = saliency_img->val[p];
  }

  /*get the biggest object component in the image*/
  largest_size = 0;
  largest_saliency = 0;
  largest_label = -1;

  for( int i = 0; i <= max_label; i++) {
    if(region_area[i] > largest_size && region_saliency[i] > largest_saliency) {
        largest_saliency = region_saliency[i];
        largest_size = region_area[i];
        largest_label = i;
    }
  }

  nlabel=iftCreateImage(label_img->xsize, label_img->ysize, label_img->zsize);

  for (int p=0;p<label_img->n;p++){
    if (rlabel->val[p] == largest_label) nlabel->val[p] = 0;
    else nlabel->val[p] = 255;
  }

  iftDestroyAdjRel(&A);
  iftDestroyImage(&rlabel);
  iftFree(region_area);
  iftFree(region_saliency);

  return nlabel;
}

iftImage *iftSupressLargestSaliencyGlobal(iftImage *saliency_img)
{
  int max_saliency;
  iftImage *thresh;

  max_saliency = iftMaximumValue(saliency_img);
  thresh = iftThreshold(saliency_img, max_saliency, max_saliency, 255);

  #pragma omp parallel for
  for(int p = 0; p < thresh->n; p++) {
    thresh->val[p] = 255 - thresh->val[p];
  }

  return thresh;
}

iftImage *iftSupressAdaptMeanSaliencyGlobal(iftImage *saliency_img)
{
  int adap_thresh, mean_saliency, max_saliency;
  iftImage *thresh;

  max_saliency = iftMaximumValue(saliency_img);

  mean_saliency = 0;
  #pragma omp parallel for reduction(+:mean_saliency)
  for(int p = 0; p < saliency_img->n; p++) mean_saliency += saliency_img->val[p];
  
  adap_thresh = iftRound(mean_saliency/(float)saliency_img->n);

  thresh = iftThreshold(saliency_img, adap_thresh, max_saliency, 255);

  #pragma omp parallel for
  for(int p = 0; p < thresh->n; p++) thresh->val[p] = 255 - thresh->val[p];

  return thresh;
}

void iftWriteOverlay
(iftImage* orig, iftImage *label, const char *filename)
{
  int normvalue;
  iftImage *overlay;
  iftAdjRel *A;
  iftColor RGB, YCbCr;

  normvalue = iftNormalizationValue(iftMaximumValue(orig)); 

  A = iftCircular(1.0);
  
  overlay = iftBorderImage(label,0);

  RGB = iftRGBColor(normvalue, 0, 0);
  YCbCr = iftRGBtoYCbCr(RGB, normvalue);
  
  iftDrawBorders(orig,overlay,A,YCbCr,A);

  iftWriteImageByExt(orig, filename);

  iftDestroyImage(&overlay);
  iftDestroyAdjRel(&A);
}