#include "ift.h"
#include "iftDISF.h"

int main(int argc, char *argv[]) 
{
  iftImage  *img, *label;
  iftMImage *mimg;
  iftImage  *gt_borders=NULL, *gt_regions=NULL, *border=NULL;
  iftAdjRel *A;
  int        nseeds;
  char       labelfilename[256];
  char       borderfilename[256];
  iftColor   RGB, YCbCr;
  int        normvalue;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=5 && argc!=7 )
    iftError("Usage: iftDISF <image.[*]> <nregions> <smooth niters (e.g., 2)> <output_label> <ground-truth border image (OPTIONAL)> <ground-truth region image (OPTIONAL)>","main");

  img  = iftReadImageByExt(argv[1]);

  if (argc == 7){
    gt_borders = iftReadImageByExt(argv[5]);
    gt_regions = iftReadImageByExt(argv[6]);
  }
  
  normvalue =  iftNormalizationValue(iftMaximumValue(img)); 

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

  int nregions = atoi(argv[2]);
  label        = runDISF(mimg, 5*nregions, nregions, A, NULL);
  border       = iftBorderImage(label,0);
  iftDestroyAdjRel(&A);
  iftDestroyMImage(&mimg);
  
  A      = iftCircular(0.0);  
  nseeds = iftMaximumValue(label);

  int smooth_niters = atoi(argv[3]);
  if (smooth_niters > 0){
    iftAdjRel *B;
    if (iftIs3DImage(img)){
      B = iftSpheric(sqrtf(3.0));
    }else{
      B = iftCircular(5.0);
    }
    iftImage  *basins  = iftImageGradientMagnitude(img,A);
    iftFImage *weight  = iftSmoothWeightImage(basins,0.5);
    iftImage  *nlabel  = iftFastSmoothObjects(label,weight,smooth_niters);
    iftDestroyAdjRel(&B);
    iftDestroyImage(&basins);
    iftDestroyFImage(&weight);
    iftDestroyImage(&label);
    label = nlabel;
  }

  // Compute metrics
  if (argc==7){
    float br, ue, comp, topology;
    br        = iftBoundaryRecall(gt_borders, border, 2.0);
    printf("BR: %f \n", br);
    
    ue       = iftUnderSegmentation(gt_regions, label);
    printf("UE: %f \n", ue);

    comp     = iftCompactness2D(label);
    printf("Comp: %f \n", comp);

    topology = iftTopologyMeasure(label);
    printf("Top: %f \n", topology);
  }
  
  // Write output image
  if (!iftIs3DImage(img)){
    if (argc == 7){
      RGB.val[0] = normvalue;
      RGB.val[1] = 0;
      RGB.val[2] = normvalue;
      YCbCr      = iftRGBtoYCbCr(RGB, normvalue);
      iftDrawBorders(img,gt_borders,A,YCbCr,A);
    }

    RGB.val[0] = normvalue;
    RGB.val[1] = normvalue/2;
    RGB.val[2] = 0;
    
    YCbCr      = iftRGBtoYCbCr(RGB, normvalue);
    sprintf(labelfilename, "%s_label.png", argv[4]);
    iftWriteImageP2(label,labelfilename);
    iftDrawBorders(img,border,A,YCbCr,A);
    sprintf(borderfilename, "%s.png", argv[4]);        
    iftWriteImageByExt(img, borderfilename);
  } else {
    printf("%d\n",iftMaximumValue(label));
    sprintf(labelfilename, "%s.nii.gz", argv[4]);
    iftWriteImage(label,labelfilename);
    RGB.val[2] = RGB.val[1] = RGB.val[0] = iftMaximumValue(img);
    YCbCr      = iftRGBtoYCbCr(RGB, RGB.val[0]);
    iftLabeledSet* Centers = iftGeodesicCenters(label);
    iftDestroyAdjRel(&A);
    A = iftSpheric(1.0);
    while (Centers != NULL) {
      int trash;
      int p = iftRemoveLabeledSet(&Centers,&trash);
      iftVoxel u = iftGetVoxelCoord(img,p);
      iftDrawPoint(img, u, YCbCr, A, normvalue);
    }
    sprintf(borderfilename, "%s-centers.nii.gz", argv[7]);
    iftWriteImage(img,borderfilename);
  }

  printf("Number of superpixels = %d \n", nseeds);
    
  // Free
  iftDestroyImage(&img);
  if (argc==7) {
    iftDestroyImage(&gt_borders);
    iftDestroyImage(&gt_regions);
  }
  iftDestroyImage(&label);
  iftDestroyImage(&border);
  iftDestroyAdjRel(&A);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
     MemDinInicial,MemDinFinal);   

  return(0);
}
