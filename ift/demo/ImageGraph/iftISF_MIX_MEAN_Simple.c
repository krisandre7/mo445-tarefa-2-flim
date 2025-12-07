#include "ift.h"

iftImage *iftMyExtract_ISF_MIX_MEAN_Superpixels(iftImage *img, iftImage *mask1, int nsuperpixels, float alpha, float beta, int niters, int smooth_niters, int *nseeds, int *finalniters) {
  iftImage  *seeds, *label;
  iftMImage *mimg;
  iftAdjRel *A;
  iftIGraph *igraph;
  timer     *t1=NULL,*t2=NULL;
  bool destroy_mask1 = false;
  
  /* Compute ISF superpixels */
  if (iftIs3DImage(img)){
    A      = iftSpheric(1.0);
  } else {
    A      = iftCircular(1.0);
  }

  if (iftIsColorImage(img)){
    mimg   = iftImageToMImage(img,LABNorm_CSPACE);
  } else {
    mimg   = iftImageToMImage(img,GRAY_CSPACE);
  }

  if (mask1 == NULL){
    mask1  = iftSelectImageDomain(mimg->xsize,mimg->ysize,mimg->zsize);
    igraph = iftImplicitIGraph(mimg,mask1,A);
    destroy_mask1 = true;
  } else {
    igraph = iftExplicitIGraph(mimg,mask1,NULL,A);    
  }
  
  
  t1 = iftTic();
  /* seed sampling for ISF */
  seeds   = iftAltMixedSampling(mimg,mask1,nsuperpixels);
  *nseeds = iftNumberOfElements(seeds);
  
  if (destroy_mask1==true)
    iftDestroyImage(&mask1);
  iftDestroyMImage(&mimg);

  *finalniters = iftIGraphISF_Mean(igraph,seeds,alpha,beta,niters);

  /* Smooth regions in the label map: only works for implicit
     graphs */
  if ((smooth_niters > 0)&&(destroy_mask1)){
    iftIGraphSetWeightForRegionSmoothing(igraph, img);
    iftIGraphSmoothRegions(igraph, smooth_niters);
  }
  label = iftIGraphLabel(igraph);

  t2 = iftToc();
  printf("ISF proc time im ms: %f\n", iftCompTime(t1,t2));

  iftDestroyImage(&seeds);
  iftDestroyIGraph(&igraph);
  iftDestroyAdjRel(&A);
  

  return label;
}

int main(int argc, char *argv[]) 
{
  iftImage  *img, *label, *mask = NULL;
  int        nautoiters, nseeds;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=8 && argc!=9 )
    iftError("Usage: iftISF <input image> <output image> <nsamples> <alpha (e.g., [0.005-0.2])> <beta (e.g., 12)> <niters (e.g., 10)> <smooth niters (e.g., 2)> < mask image (OPTIONAL)>","main");

  if (argc == 9)
    mask = iftReadImageByExt(argv[8]);
  
  img       = iftReadImageByExt(argv[1]);
  label     = iftMyExtract_ISF_MIX_MEAN_Superpixels(img, mask, atoi(argv[3]), atof(argv[4]), atof(argv[5]), atoi(argv[6]), atoi(argv[7]), &(nseeds), &(nautoiters));

  iftWriteImageByExt(label,argv[2]);    

  // Print number of iterations and superpixels
  printf("Number of iterations = %d \n", nautoiters);
  printf("Number of superpixels = %d \n", nseeds);
    
  // Free
  iftDestroyImage(&img);
  iftDestroyImage(&label);
  if (argc == 9)
    iftDestroyImage(&mask);
  
  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
     MemDinInicial,MemDinFinal);   

  return(0);
}
