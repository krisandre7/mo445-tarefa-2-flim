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
    iftError("Usage: iftInfRec <image.*> <seeds.txt>","main");

  img[0]   = iftReadImageByExt(argv[1]);    
  iftLabeledSet *S = iftReadSeeds(img[0],argv[2]);
  
  t1     = iftTic();

  iftImage *marker = iftCreateImageFromImage(img[0]);
  iftSetImage(marker,IFT_INFINITY_INT_NEG);
  while (S != NULL){
    int label;
    int p = iftRemoveLabeledSet(&S,&label);
    marker->val[p] = img[0]->val[p];
  }
  
  img[1] = iftInferiorRec(img[0],marker,NULL);

  t2     = iftToc();
  fprintf(stdout,"InferiorRec executed in %f ms\n",iftCompTime(t1,t2));

  if (iftIs3DImage(img[0])){
      iftWriteImageByExt(img[1],"result.scn");
  } else {
    if (iftIsColorImage(img[1]))
      iftWriteImageByExt(img[1],"result.png");
    else
      iftWriteImageByExt(img[1],"result.pgm");
  }

  iftDestroyImage(&img[0]);  
  iftDestroyImage(&img[1]);  
  iftDestroyImage(&marker);  

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}

