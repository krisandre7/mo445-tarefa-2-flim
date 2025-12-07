#include "ift.h"

#define GRAYSPACE  GRAY_CSPACE 
#define COLORSPACE YCbCr_CSPACE 

iftDataSet *iftTestSet(iftImage *img)
{
  iftMImage  *mimg;
  iftDataSet *Z;
  
  if (!iftIsColorImage(img)) {
    mimg = iftImageToMImage(img, GRAYSPACE);
  }
  else{
    mimg = iftImageToMImage(img, COLORSPACE);
  }
  
  Z = iftMImageToDataSet(mimg, NULL);

  return(Z);
}

iftDataSet *iftTrainingSet(iftImage *img, int nsamples)
{
  iftMImage  *mimg;
  iftDataSet *Z;
  
  if (!iftIsColorImage(img)) {
    mimg = iftImageToMImage(img, GRAYSPACE);
  }
  else{
    mimg = iftImageToMImage(img, COLORSPACE);
  }
  
  iftImage *mask1 = iftSelectImageDomain(img->xsize,img->ysize,img->zsize);
  iftImage *mask2 = iftGridSampling(mimg,mask1,nsamples);
  Z               = iftMImageToDataSetInRegion(mimg,mask2);
  iftSetStatus(Z,IFT_TRAIN);
  iftDestroyImage(&mask1);
  iftDestroyImage(&mask2);
  
  return(Z);
}

int main(int argc, char *argv[])
{
  iftImage        *img=NULL, *pdf=NULL;
  iftDataSet      *Ztrain=NULL, *Z=NULL;
  iftKnnGraph     *graph=NULL;
  timer           *t1=NULL,*t2=NULL;

  /* It creates the pdf of the input image and the groups of its domes */
  
  if (argc != 6)
    iftError("Usage: iftImagePDF <input_image.[ppm,png,pgm]> <number of samples> <kmax> <output_pdf.[pgm,png]> <output_groups.[ppm,png]","main");

  t1=iftTic();

  img    = iftReadImageByExt(argv[1]);
  Ztrain = iftTrainingSet(img,atoi(argv[2]));
  Z      = iftTestSet(img);
  if (iftIsColorImage(img)){
    Ztrain->alpha[0]=0.2; Ztrain->alpha[1]=0.4; Ztrain->alpha[2]=0.4;
  }
  graph  = iftCreateKnnGraph(Ztrain,atoi(argv[3]));

  iftUnsupTrain(graph,iftNormalizedCut);
  iftUnsupClassify(graph,Z);

  t2=iftToc();
  fprintf(stdout,"Computational time was %.2fms\n",iftCompTime(t1,t2));

  pdf = iftDataSetWeight(Z);
  iftWriteImageByExt(pdf,argv[4]);

  iftImage *label    = iftDataSetToLabelImage(Z,0);
  iftImage *clusters = iftColorizeComp(label);
  iftWriteImageByExt(clusters,argv[5]);

  iftDestroyImage(&label);
  iftDestroyImage(&clusters);
  iftDestroyImage(&img);
  iftDestroyImage(&pdf);
  iftDestroyDataSet(&Ztrain);
  iftDestroyDataSet(&Z);
  iftDestroyKnnGraph(&graph);
  return(0);
}




