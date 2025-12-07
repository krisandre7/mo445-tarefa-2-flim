#include "ift.h"

int main(int argc, char *argv[]) 
{

  if (argc!=3)
    iftError("Usage: iftAlignObjectByPCA <binary.*> <aligned.*>","main");

  timer *t1=iftTic();
  
  iftImage *bin  = iftReadImageByExt(argv[1]);
  if (!iftIsBinaryImage(bin)){
    iftImage *aux= iftThreshold(bin,127,255,255);
    iftDestroyImage(&bin);
    bin = aux;
  }
  
  iftImage *rbin = iftAlignObject(bin);

  iftWriteImageByExt(rbin,argv[2]); 
  
  iftDestroyImage(&rbin);
  iftDestroyImage(&bin);
  
  timer *t2=iftToc(); 
  printf("%f ms\n",iftCompTime(t1,t2)); 
 
  return(0);
}

