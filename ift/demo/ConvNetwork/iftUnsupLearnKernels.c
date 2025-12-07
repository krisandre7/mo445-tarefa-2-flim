#include "ift.h"

#define NLAYERS 1

int main(int argc, char **argv) 
{
  iftImage       *orig;
  iftMImage      *input[NLAYERS], *output[NLAYERS];
  char            filename[200];
  char            ext[10],*pos;
  timer          *t1=NULL,*t2=NULL;
  iftConvNetwork *convnet;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/


  if (argc!=3)
    iftError("Usage: iftUnsupLearnKernels <input.[ppm,pgm]> <input_parameters.convnet> ","main");

  pos = strrchr(argv[1],'.') + 1;
  sscanf(pos,"%s",ext);
  
  if (strcmp(ext,"ppm")==0){
    orig   = iftReadImageP6(argv[1]);    
  }else{
    if (strcmp(ext,"pgm")==0){
      orig   = iftReadImageP5(argv[1]);    
    }else{
      printf("Invalid image format: %s\n",ext);
      exit(-1);
    }
  }

  pos = strrchr(argv[2],'.') + 1;
  sscanf(pos,"%s",ext);

  iftRandomSeed(IFT_RANDOM_SEED);

  if (strcmp(ext,"convnet")==0){
    convnet = iftReadConvNetwork(argv[2]);
  }else{
    printf("Invalid file format: %s\n",ext);
    exit(-1);
  }

  /* Compute kernels by unsupervised learning */

  input[0]  = iftImageToMImage(orig,WEIGHTED_YCbCr_CSPACE);
  iftWriteMImage(input[0],"input.mig");
  
  t1     = iftTic(); 
  for (int l=0; l < NLAYERS; l++) {
    if (l>0) {
      input[l] = output[l-1];
    }
    
    iftUnsupLearnKernels(input[l],convnet,2000,0.1,1);

    output[l] = iftApplyConvNetwork(input[l],convnet);  
    sprintf(filename,"output-%d.mig",l);
    iftWriteMImage(output[l],filename);    
    convnet->input_norm_adj_param=0.0;
    convnet->input_xsize=output[l]->xsize;
    convnet->input_ysize=output[l]->ysize;
    convnet->input_zsize=output[l]->zsize;
    convnet->input_nbands=output[l]->m;
    sprintf(filename,"output-layer-%d.convnet",l);
    iftWriteConvNetwork(convnet,filename);  
    iftDestroyConvNetwork(&convnet);
    convnet = iftReadConvNetwork(filename);
  }
  t2     = iftToc();
  fprintf(stdout,"Kernels learned in %f ms\n",iftCompTime(t1,t2));

  /* Write features */

  iftWriteMImage(output[NLAYERS-1],"output.mig");
  
  t1 = iftTic();
  for (int i=0; i < convnet->nkernels[convnet->nlayers-1]; i++) {
    iftImage *img = iftMImageToImage(output[NLAYERS-1],255,i);
    if (img != NULL){
      sprintf(filename,"feat%d.pgm",i);
      iftWriteImageP5(img,filename);
      iftDestroyImage(&img);
    }
  }
  t2     = iftToc();
  fprintf(stdout,"Output features written in %f ms\n",iftCompTime(t1,t2));
  

  /* Compute basins and watershed from gray marker */

  t1 = iftTic();

  iftAdjRel *A=iftCircular(sqrtf(2.0));  
  iftImage *basins=iftMImageBasins(output[NLAYERS-1],A);
  iftWriteImageP2(basins,"basinsByCNN.pgm");

  iftImage *marker = iftVolumeClose(basins,100);
  iftImage *label  = iftWaterGray(basins,marker,A);
  iftDestroyImage(&marker);
  iftDestroyImage(&basins);
  iftColor RGB, YCbCr;
  RGB.val[0] = 255; RGB.val[1] = 0; RGB.val[2] = 0;
  YCbCr            = iftRGBtoYCbCr(RGB);
  iftAdjRel *B     = iftCircular(0.0);
  int         size = (input[0]->xsize - output[NLAYERS-1]->xsize)/2;
  iftImage *watergray    = iftRemFrame(orig,size);
  free(watergray->Cb);
  free(watergray->Cr);
  watergray->Cb = watergray->Cr = NULL;
  iftDrawBorders(watergray,label,A,YCbCr,B);
  iftWriteImageP6(watergray,"WatergrayByCNN.ppm");
  iftDestroyImage(&label);
  iftDestroyImage(&watergray);
  iftDestroyAdjRel(&A);
  iftDestroyAdjRel(&B);
  t2     = iftToc();
  fprintf(stdout,"Watergray computed in %f ms\n",iftCompTime(t1,t2));

  iftDestroyImage(&orig);
  iftDestroyConvNetwork(&convnet);
  for (int l=0; l < NLAYERS; l++) 
    iftDestroyMImage(&output[l]);

  iftDestroyMImage(&input[0]);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   


  return(0);
}
