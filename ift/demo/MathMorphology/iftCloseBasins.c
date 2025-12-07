#include "ift.h"

int main(int argc, char *argv[]) 
{
  iftImage       *img[2];
  timer          *t1=NULL,*t2=NULL;

  if (argc!=3)
    iftError("Usage: iftCloseBasins <image.[pgm,scn,ppm]> <out.[pgm,scn,ppm]>","main");

  img[0]   = iftReadImageByExt(argv[1]);    

  t1     = iftTic();


  img[1] = iftCloseBasins(img[0],NULL,NULL);

  t2     = iftToc();
  fprintf(stdout,"CloseBasins executed in %f ms\n",iftCompTime(t1,t2));

  iftWriteImageByExt(img[1], argv[2]);

  iftDestroyImage(&img[0]);  
  iftDestroyImage(&img[1]);  


  return(0);
}

