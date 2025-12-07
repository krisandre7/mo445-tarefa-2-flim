#include "ift.h"

int main(int argc, char *argv[]) 
{
  iftImage        *img=NULL,*label=NULL,*basins;
  iftImage        *marker=NULL;
  iftColor         RGB,YCbCr;
  iftAdjRel       *A=NULL,*B=NULL;
  iftCplGraph     *graph=NULL;
  iftLabeledSet   *seed=NULL;
  iftDataSet      *Z=NULL;
  char             ext[10],*pos;
  timer           *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/


  if (argc!=5)
    iftError("Usage: iftClassifyImageComps <image.[pgm,ppm]> <spatial_radius> <volume_thres> <seeds.txt>","main");

  iftRandomSeed(IFT_RANDOM_SEED);

  pos = strrchr(argv[1],'.') + 1;
  sscanf(pos,"%s",ext);
  
  if (strcmp(ext,"ppm")==0){
    img   = iftReadImageP6(argv[1]);    
  }else{
    if (strcmp(ext,"pgm")==0){
      img   = iftReadImageP5(argv[1]);    
    }else{
      printf("Invalid image format: %s\n",ext);
      exit(-1);
    }
  }

  pos = strrchr(argv[4],'.') + 1;
  sscanf(pos,"%s",ext);

  if (strcmp(ext,"txt")==0){
    seed=iftReadSeeds2D(argv[4],img);
  }else{
    printf("Invalid file format: %s\n",ext);
    exit(-1);
  }

  
  t1 = iftTic();


  /* the operation is connected for the topology defined by A: A must
     be the same in all operators (including
     iftVolumeClose?). Otherwise, this is not a connected operation in
     the desired topology. */

  A      = iftCircular(atof(argv[2]));
  basins = iftImageBasins(img,A);
  iftDestroyAdjRel(&A);

  marker = iftVolumeClose(basins,atof(argv[3]));
  A      = iftCircular(atof(argv[2]));
  label  = iftWaterGray(basins,marker,A);
  Z      = iftSupervoxelsToDataSet(img,label);
  if (Z->nfeats==3) {
    Z->alpha[0]=0.2;
    Z->alpha[1]=1.0;
    Z->alpha[2]=1.0;
  }

  /* Specify training samples */
  iftLabeledSet *S=seed;
  int s,p;
  Z->nclasses = 0;
  while(S != NULL) {
    p = S->elem;
    s = label->val[p]-1;
    Z->sample[s].truelabel  = S->label+1;
    Z->sample[s].status = IFT_TRAIN;
    if (Z->sample[s].truelabel > Z->nclasses)
      Z->nclasses = Z->sample[s].truelabel;
    S = S->next;
  }

  Z->ntrainsamples = 0;
  for (s=0; s < Z->nsamples; s++) 
    if (Z->sample[s].status == IFT_TRAIN) 
      Z->ntrainsamples++;


  graph  = iftCreateCplGraph(Z);              // Training
  iftSupTrain(graph);
  iftClassify(graph,Z);                      // Classify test samples 

  iftImage *bin=iftCreateImage(img->xsize,img->ysize,img->zsize);

  for (p=0; p < img->n; p++) {
    s = label->val[p]-1;
    bin->val[p] = Z->sample[s].label-1;
  }

  iftImage *segm = iftMask(img,bin);
  iftWriteImageP6(segm,"segm.ppm");
  iftDestroyImage(&segm);

  t2     = iftToc(); 

  fprintf(stdout,"watergray in %f ms with %d regions\n",iftCompTime(t1,t2),iftMaximumValue(label));

  iftDestroyAdjRel(&A);
  iftDestroyDataSet(&Z);
  iftDestroyCplGraph(&graph);

  iftWriteImageP2(label,"labels.pgm");
  //  iftWriteImageP2(basins,"basins.pgm");

  RGB.val[0] = 255;
  RGB.val[1] = 0;
  RGB.val[2] = 0;
  YCbCr      = iftRGBtoYCbCr(RGB);
  A          = iftCircular(sqrtf(2.0));
  B          = iftCircular(0.0);
  iftDrawBorders(img,label,A,YCbCr,B);
  iftWriteImageP6(img,"result.ppm");

  iftDestroyAdjRel(&A);
  iftDestroyAdjRel(&B);
  iftDestroyImage(&img);  
  iftDestroyImage(&marker);  
  iftDestroyImage(&basins);  
  iftDestroyImage(&label);  
  iftDestroyImage(&bin);  
  iftDestroyLabeledSet(&seed);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}

