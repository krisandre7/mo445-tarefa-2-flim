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
    iftError("Usage: iftAreaClose <image.[pgm,ppm,scn]> <area_thres>","main");


  img[0]   = iftReadImageByExt(argv[1]);    
  
  t1     = iftTic();

  img[1] = iftAreaClose(img[0],atoi(argv[2]), NULL);
  // img[1] = iftFastAreaClose(img[0],atoi(argv[2]));

  t2     = iftToc();
  fprintf(stdout,"Area closing executed in %f ms\n",iftCompTime(t1,t2));

  if (iftIs3DImage(img[0])){
      iftWriteImageByExt(img[1],"result.scn");
  } else {
    if (iftIsColorImage(img[1]))
      iftWriteImageByExt(img[1],"result.ppm");
    else
      iftWriteImageByExt(img[1],"result.pgm");
  }

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

