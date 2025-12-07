#include "ift.h"


int main(int argc, char **argv) 
{
  iftMImage      *input;
  iftImage       *output;
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


  if (argc!=6)
    iftError("Usage: iftCombineFeatures <input.mig> <feat R> <feat G> <feat B> <output.ppm>","main");

  pos = strrchr(argv[1],'.') + 1;
  sscanf(pos,"%s",ext);

  if (strcmp(ext,"mig")==0){
    input = iftReadMImage(argv[1]);
  }else{
    printf("Invalid file format: %s\n",ext);
    exit(-1);
  }

  t1 = iftTic();
  iftImage *R = iftMImageToImage(input,255,atoi(argv[2]));
  iftImage *G = iftMImageToImage(input,255,atoi(argv[3]));
  iftImage *B = iftMImageToImage(input,255,atoi(argv[4]));
  output      = iftCreateImage(input->xsize,input->ysize,input->zsize);
  iftSetCbCr(output,128);
  for (int p=0; p < input->n; p++) {
    iftColor RGB;
    RGB.val[0] = R->val[p];
    RGB.val[1] = G->val[p];
    RGB.val[2] = B->val[p];
    iftColor YCbCr = iftRGBtoYCbCr(RGB);
    output->val[p] = YCbCr.val[0];
    output->Cb[p] = YCbCr.val[1];
    output->Cr[p] = YCbCr.val[2];
  }
  t2 = iftToc();
  fprintf(stdout,"Image combination in %f ms\n",iftCompTime(t1,t2));

  iftWriteImageP6(output,argv[5]);

  iftDestroyMImage(&input);
  iftDestroyImage(&output);
  iftDestroyImage(&R);
  iftDestroyImage(&G);
  iftDestroyImage(&B);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}
