#include "ift.h"

/* It would be interesting to build a single dictionary for the entire
   image database, so the quantitized colors would be the same for
   robust segmentation. It might be a good idea to select each
   connected component of the same color, close its holes, and
   evaluate if it is a parasite. Would the color-quantitized image
   facilitate the deep learning process? */

typedef struct _ift_BoVW {
  iftColor *YCbCr;
  int       n, m;
} iftBoVW;

iftBoVW *iftCreateBoVW(iftKnnGraph *graph)
{
  iftBoVW *bovw = (iftBoVW *) calloc(1,sizeof(iftBoVW));

  bovw->n   = graph->Z->nlabels;
  bovw->m   = graph->Z->nfeats;
  bovw->YCbCr = (iftColor *) calloc(bovw->n,sizeof(iftColor));

  for (int i=0, j=0; i < graph->nnodes; i++)
    if (i==graph->node[i].root){
      int s = graph->node[i].sample;
      for (int k=0; k < graph->Z->nfeats; k++) 
	bovw->YCbCr[j].val[k]=graph->Z->sample[s].feat[k];
      j++;
    }

  /* for (int i=0; i < bovw->n; i++){ */
  /*   for (int j=0; j < bovw->m; j++) */
  /*     printf("%d ",bovw->YCbCr[i].val[j]); */
  /*   printf("\n"); */
  /* } */

  return(bovw);
}

int iftClosestCodeWord(iftImage *img, int p, iftBoVW *bovw)
{
  float dist, mindist;
  int   closest;

  mindist = 0.0; closest = 0;
  if (bovw->m==1){
    mindist = abs(img->val[p]-bovw->YCbCr[closest].val[0]);
    for (int i=1; i < bovw->n; i++){
      dist = abs(img->val[p]-bovw->YCbCr[i].val[0]);
      if (dist < mindist){
	mindist = dist;
	closest = i;
      }
    }	
  }else{
    mindist = abs(img->val[p]-bovw->YCbCr[closest].val[0]) + 
      abs(img->Cb[p]-bovw->YCbCr[closest].val[1]) +
      abs(img->Cr[p]-bovw->YCbCr[closest].val[2]);
    for (int i=1; i < bovw->n; i++){
      dist = abs(img->val[p]-bovw->YCbCr[i].val[0]) +
	abs(img->Cb[p]-bovw->YCbCr[i].val[1]) +
	abs(img->Cr[p]-bovw->YCbCr[i].val[2]);
      if (dist < mindist){
	mindist = dist;
	closest = i;
      }
    }	
  }
   
  return(closest);    
}

iftDataSet *iftImageToDataSetBoVW(iftImage *img, iftBoVW *bovw)
{
  iftDataSet *Z;

  Z=iftCreateDataSet(img->n,bovw->m);
  for (int p=0; p < img->n; p++) {
    int i = iftClosestCodeWord(img,p,bovw);
    Z->sample[p].label = i+1;
    for (int k=0; k < bovw->m; k++)
      Z->sample[p].feat[k] = bovw->YCbCr[i].val[k];
    Z->sample[p].id              = p;
  }
  
  Z->ref_data = img;
  Z->ref_data_type = IFT_REF_DATA_IMAGE;
  if (iftIsColorImage(img)){
    Z->alpha[0]=0.2;
    Z->alpha[1]=0.4;
    Z->alpha[2]=0.4;
  }
  
  Z->ref_data = img;
  Z->ref_data_type = IFT_REF_DATA_IMAGE;

  return(Z);
}

iftImage *iftDrawCodeWords(iftImage *img, iftKnnGraph *graph)
{
  iftImage   *cwords = iftCopyImage(img);
  iftDataSet *Z = graph->Z;
  iftAdjRel  *A = iftCircular(3.0);

  for (int i=0; i < Z->nsamples; i++){
    if (graph->node[i].root == i){
      int s = graph->node[i].sample;
      iftColor RGB, YCbCr;
      RGB.val[0]=255;
      RGB.val[1]=255;
      RGB.val[2]=0;
      YCbCr = iftRGBtoYCbCr(RGB,255);
      iftDrawPoint(cwords,iftGetVoxelCoord(cwords,Z->sample[s].id),YCbCr,A);
    }
  }
  iftDestroyAdjRel(&A);
  
  return(cwords);
}

iftDataSet *iftTrainingSet(iftImage *img, int nsamples)
{
  iftMImage  *mimg;
  iftDataSet *Z;
  
  if (!iftIsColorImage(img)) {
    mimg = iftImageToMImage(img, GRAY_CSPACE);
  }
  else{
    mimg = iftImageToMImage(img, YCbCr_CSPACE);
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
  iftImage        *img=NULL, *cwords=NULL;
  iftDataSet      *Z=NULL;
  iftKnnGraph     *graph=NULL;
  iftBoVW         *bovw=NULL;
  timer           *t1=NULL,*t2=NULL;

  if (argc != 5)
    iftError("Usage: iftColorQuantization <image.[ppm,png,pgm]> <nsamples (e.g., 2000)> <kmax (e.g., 20)> <result.[ppm,png,pgm]","main");

  t1=iftTic();

  iftRandomSeed(IFT_RANDOM_SEED);

  //  img   = iftReadPNGImage(argv[1]);
  img   = iftReadImageByExt(argv[1]);
  printf("%d\n",iftMaximumValue(img));

  Z    = iftTrainingSet(img,atoi(argv[2]));
  if (iftIsColorImage(img)){
    Z->alpha[0]=0.2;
    Z->alpha[1]=0.4;
    Z->alpha[2]=0.4;
  }
  graph = iftCreateKnnGraph(Z,atoi(argv[3]));

  iftUnsupTrain(graph,iftNormalizedCut);
  
  printf("number of code words %d\n",Z->nlabels);
  
  cwords = iftDrawCodeWords(img,graph);
  iftWriteImageByExt(cwords,"cwords.png");
  iftDestroyImage(&cwords);

  bovw = iftCreateBoVW(graph);
  iftMImage *mimg = (iftMImage *)Z->ref_data;
  iftDestroyMImage(&mimg);
  iftDestroyDataSet(&Z);
  iftDestroyKnnGraph(&graph);

  Z = iftImageToDataSetBoVW(img, bovw);

 
  iftImage *aux1  = iftDataSetToLabelImage(Z,0);
  iftImage *aux2  = iftSmoothRegionsByDiffusion(aux1,img,0.5,5);
  iftDestroyImage(&aux1);
  iftImage *label = iftColorizeComp(aux2);
  iftWriteImageByExt(label,"label.png");
  iftDestroyImage(&label);
  iftDestroyImage(&aux2);

  
  iftImage *res = iftCopyImage(img);
  if (iftIsColorImage(img)){
    for (int p=0; p < img->n; p++){
      res->val[p] = bovw->YCbCr[Z->sample[p].label-1].val[0];
      res->Cb[p]  = bovw->YCbCr[Z->sample[p].label-1].val[1];
      res->Cr[p]  = bovw->YCbCr[Z->sample[p].label-1].val[2];
    }
  } else {
    for (int p=0; p < img->n; p++){
      res->val[p] = bovw->YCbCr[Z->sample[p].label-1].val[0];
    }
  }
  iftWriteImageByExt(res,argv[4]);
  iftDestroyImage(&res);

  
  t2=iftToc();
  fprintf(stdout,"Computational time was %.2fms\n",iftCompTime(t1,t2));

  iftDestroyKnnGraph(&graph);
  iftDestroyImage(&img);
  iftDestroyDataSet(&Z);

  return(0);
}




