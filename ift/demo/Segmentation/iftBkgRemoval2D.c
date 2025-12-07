#include <ift.h>

int main(int argc, char *argv[]) 
{
  iftImage        *img=NULL,*label=NULL,*dens=NULL,*aux=NULL;
  iftDataSet      *Z=NULL, *Z1=NULL;
  iftKnnGraph     *graph=NULL;
  iftColor         RGB,YCbCr;
  iftAdjRel       *A=NULL,*B=NULL;
  int              s,p; 
  iftVoxel         u;
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


  if (argc!=4)
    iftError("Usage: iftBkgRemoval2D <image.ppm> <size of the border> <number of samples>","main");


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

  Z      = iftImageToDataSet(img);
  Z1     = iftImageBorderDataSet(Z, atoi(argv[2]), atoi(argv[3]));
  iftSetDistanceFunction(Z1, 1);
  if (Z1->nfeats==3){
    Z1->alpha[0] = 0.20;
    Z1->alpha[1] = 1.00;
    Z1->alpha[2] = 1.00;
  }

  t1 = iftTic();

  iftSelectUnsupTrainSamples(Z1,0.2,10);
  graph=iftCreateKnnGraph(Z1,0.5);
  iftUnsupTrain(graph,iftNormalizedCut);
  printf("number of outliers in the test set is %d\n",iftUnsupClassify(graph,Z));

  t2     = iftToc(); 

  fprintf(stdout,"clustering in %f ms with %d groups\n",iftCompTime(t1,t2),Z->ngroups);

  dens = iftDataSetWeight(Z);
  iftWriteImageP2(dens,"weight.pgm");
  aux = iftBinarizeByOPF(img,dens,iftOtsu(dens),0.01);
  //aux = iftThreshold(dens,iftOtsu(dens),INFINITY_INT,255);
  //label   = iftFastAreaClose(aux,400);
  label = iftCloseBasins(aux,NULL,NULL);
  iftDestroyImage(&aux);

  iftWriteImageP5(label,"labels.pgm");

  A          = iftCircular(sqrtf(2.0));

  RGB.val[0] = 255;
  RGB.val[1] = 0;
  RGB.val[2] = 0;
  YCbCr      = iftRGBtoYCbCr(RGB,255);
  B          = iftCircular(3.0);
  aux        = iftCopyImage(img);
  for (s=0; s < graph->nnodes; s++) {
    p = graph->Z->sample[graph->node[s].sample].id;
    u = iftGetVoxelCoord(img,p);
    iftDrawPoint(aux,u,YCbCr,B,255);
  }
  iftWriteImageP6(aux,"samples.ppm");
  iftDestroyImage(&aux);
  iftDestroyAdjRel(&B);

  B          = iftCircular(0.0);
  RGB.val[0] = 255;
  RGB.val[1] = 0;
  RGB.val[2] = 0;
  YCbCr      = iftRGBtoYCbCr(RGB,255);
  aux        = iftCopyImage(img);
  iftImage *flabel = iftFastAreaOpen(label,100);

  iftDrawBorders(aux,flabel,A,YCbCr,B);
  iftWriteImageP6(aux,"regions.ppm");
  iftDestroyImage(&aux);
  iftDestroyAdjRel(&B);
  iftDestroyImage(&flabel);

  RGB.val[0] = 255;
  RGB.val[1] = 0;
  RGB.val[2] = 0;
  YCbCr      = iftRGBtoYCbCr(RGB,255);
  B          = iftCircular(3.0);
  aux        = iftCopyImage(dens);
  float maxval = iftMaximumValue(dens);
  for (p=0; p < dens->n; p++) 
    aux->val[p] = (int) (((float)dens->val[p]/maxval)*255);
  for (s=0; s < graph->nnodes; s++) {
    if (graph->node[s].root==s){
      p = graph->Z->sample[graph->node[s].sample].id;
      u = iftGetVoxelCoord(img,p);
      iftDrawPoint(aux,u,YCbCr,B,255);
    }
  }
  iftWriteImageP6(aux,"maxima.ppm");
  iftDestroyImage(&aux);
  iftDestroyAdjRel(&B);

  iftDestroyAdjRel(&A);
  iftDestroyImage(&img);  
  iftDestroyImage(&dens);  
  iftDestroyImage(&label);  
  iftDestroyDataSet(&Z);
  iftDestroyDataSet(&Z1);
  iftDestroyKnnGraph(&graph);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}




