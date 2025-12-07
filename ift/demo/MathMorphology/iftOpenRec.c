#include "ift.h"

int main(int argc, char *argv[]) 
{
  iftImage       *img[2];
  iftAdjRel      *A;
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
    iftError("Usage: iftOpenRec <image.[pgm,scn,ppm]> <radius>","main");

  img[0]   = iftReadImageByExt(argv[1]);    

  
  t1     = iftTic();

  if (iftIs3DImage(img[0]))
    A      = iftSpheric(atof(argv[2]));
  else
    A      = iftCircular(atof(argv[2]));

  img[1] = iftOpenRec(img[0],A,NULL);

  t2     = iftToc();
  fprintf(stdout,"OpenRec executed in %f ms\n",iftCompTime(t1,t2));

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
  iftDestroyAdjRel(&A);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}

