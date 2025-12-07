#include "ift.h"


int main(int argc, char *argv[]) 
{
  iftImage       *img[3];
  timer          *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/


  if (argc!=3)
    iftError("Usage: iftEnhanceParasites <input image> <saliency map>","main");

  img[0]   = iftReadImageByExt(argv[1]);    
 
  t1     = iftTic();
  
  img[1]   = iftImageGreen(img[0]);
  img[2]   = iftImageBlue(img[0]);

  iftWriteImageByExt(img[1],"red.png");
  iftWriteImageByExt(img[2],"blue.png");
  iftDestroyImage(&img[0]);
  img[0]   = iftCreateImageFromImage(img[1]);
  
  for (int p=0; p < img[1]->n; p++){
    float value = ((float)img[1]->val[p]-(float)img[2]->val[p])/
      ((float)img[1]->val[p]+(float)img[2]->val[p]);
    if (value > 0)
      img[0]->val[p] = iftRound(65535*value);      
  }
  iftDestroyImage(&img[1]);
  iftAdjRel *A = iftCircular(3.0);
  img[1] = iftMedianFilter(img[0], A);
  iftDestroyAdjRel(&A);

  t2     = iftToc();
  fprintf(stdout,"Enhanced image in %f ms\n",iftCompTime(t1,t2));

  iftWriteImageByExt(img[1],argv[2]);
  iftDestroyImage(&img[0]);
  iftDestroyImage(&img[1]);
  iftDestroyImage(&img[2]);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}

