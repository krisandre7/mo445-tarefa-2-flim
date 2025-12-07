#include "ift.h"

iftImage *iftSegmBones(iftImage *img)
{
    puts("Initialization");
  iftImage     *bin=NULL, *bones=NULL;
  int max_val = iftMaximumValue(img);
  iftHist *hist = iftCalcGrayImageHist(img,NULL,max_val+1,max_val,true);
  int           i, thres; 
  iftAdjRel    *A;

  /* Compute accumulative histogram backwards and detect threshold at
     2.0% of it */
    puts("Compute accumulative histogram");
  thres = 0;
  for (i=max_val; (i >= 0) && (thres==0) ; i--) {
    hist->val[i] = hist->val[i+1] + hist->val[i];
    if (hist->val[i] > 0.02) //0.02
      thres = i;
  }

  iftDestroyHist(&hist);

  /* Binarize image */

    puts("Binarization");
  bin = iftThreshold(img,thres,IFT_INFINITY_INT,1);

  /* Select largest component */

    puts("Selection of largest component");
  A     = iftSpheric(sqrtf(2.0)); //3.0
  bones = iftSelectLargestComp(bin,A);
  iftDestroyImage(&bin);
  iftDestroyAdjRel(&A);

  return(bones);
}

int main(int argc, char *argv[]) 
{
  iftImage  *orig, *bones;
  timer     *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=3)
    iftError("Usage: iftSegmBones <thorax.scn> <bones.scn>","main");
  
  t1       = iftTic();

  orig     = iftReadImageByExt(argv[1]);
  //bones    = iftThreshold(orig,0.9*iftMaximumValue(orig),IFT_INFINITY_INT,255);

  bones	    = iftSegmBones(orig);

  iftImage *close = iftCloseBin(bones,3.);
  iftDestroyImage(&bones);

  iftWriteImageByExt(close,argv[2]);
  iftDestroyImage(&close);
  iftDestroyImage(&orig);

  t2     = iftToc();
  fprintf(stdout,"Bones segmented in %f ms\n",iftCompTime(t1,t2));


  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);

}
