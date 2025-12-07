#include "ift.h"

int main(int argc, char *argv[]) 
{
  iftImage       *img[2];
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
    iftError("Usage: iftOpenDomes <image.[pgm,scn,ppm]> <out.[pgm,scn,ppm]","main");


  img[0]   = iftReadImageByExt(argv[1]);    

  t1     = iftTic();

  img[1] = iftOpenDomes(img[0],NULL,NULL);

  t2     = iftToc();
  fprintf(stdout,"OpenDomes executed in %f ms\n",iftCompTime(t1,t2));

  iftWriteImageByExt(img[1], argv[2]);

  iftDestroyImage(&img[0]);  
  iftDestroyImage(&img[1]);  

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}

