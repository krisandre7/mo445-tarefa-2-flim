#include "ift.h"



int main(int argc, char **argv) 
{
  iftImage       *img,   *orig;
  iftMImage      *input, *output;
  char            ext[10],*pos;
  timer          *t1=NULL,*t2=NULL;
  iftMSConvNetwork *msconvnet;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/


  if (argc!=3)
    iftError("Usage: iftTestMSConvNetwork <image.[ppm,pgm]> <parameters.msconvnet>","main");

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

  if (strcmp(ext,"msconvnet")==0){
    msconvnet = iftReadMSConvNetwork(argv[2]);
  }else{
    printf("Invalid file format: %s\n",ext);
    exit(-1);
  }

  input  = iftImageToMImage(orig,YCbCr_CSPACE);
  t1     = iftTic();
  output = iftApplyMSConvNetwork(input,msconvnet);
  t2     = iftToc();
  fprintf(stdout,"Convolution network executed in %f ms\n",iftCompTime(t1,t2));
  printf("max: %f\n",MAXWEIGHT);
  /* New code for John */

  //msconvnet->convnet[0]->with_weights = 1;
  //iftWriteConvNetwork(msconvnet,"teste1.msconvnet");

  char filename[200];

  for (int i=0; i < msconvnet->convnet[msconvnet->nscales-1]->nkernels[0]; i++) {
    img = iftMImageToImage(output,255,i);
    sprintf(filename,"feat%d.pgm",i);
    iftWriteImageP5(img,filename);
    iftDestroyImage(&img);    
  }

  iftAdjRel *A = iftCircular(sqrtf(2.0));
  img = iftMImageBasins(output, A);
  iftWriteImageP2(img,"basins.pgm");

  iftImage *marker = iftVolumeClose(img,1000);
  iftImage *label  = iftWaterGray(img,marker,A);
  iftColor RGB, YCbCr;

  RGB.val[0] = 255;
  RGB.val[1] = 0;
  RGB.val[2] = 0;
  YCbCr      = iftRGBtoYCbCr(RGB);
  iftAdjRel *B = iftCircular(0.0);

  int sz = (orig->xsize - img->xsize)/2;
  iftDestroyImage(&img);    
  img        = iftRemFrame(orig,sz);
  iftDrawBorders(img,label,A,YCbCr,B);
  iftWriteImageP6(img,"result-new.ppm");

  iftDestroyImage(&img);
  img = iftImageBasins(orig, A);
  iftDestroyImage(&marker);    
  iftDestroyImage(&label);

  marker = iftVolumeClose(img,1000);
  label  = iftWaterGray(img,marker,A);

  RGB.val[0] = 0;
  RGB.val[1] = 0;
  RGB.val[2] = 255;
  YCbCr      = iftRGBtoYCbCr(RGB);

  iftDrawBorders(orig,label,A,YCbCr,B);
  iftWriteImageP6(orig,"result-old.ppm");



  iftDestroyImage(&img);    
  iftDestroyImage(&orig);    
  iftDestroyImage(&marker);    
  iftDestroyImage(&label);
  iftDestroyAdjRel(&A);
  iftDestroyAdjRel(&B);

  /* end of John's code */

  iftDestroyMImage(&input);
  iftDestroyMImage(&output);
  iftDestroyMSConvNetwork(&msconvnet);

 
  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   


  return(0);

}
