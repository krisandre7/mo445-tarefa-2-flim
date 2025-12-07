#include "ift.h"


iftImage *iftShapeRestoration(iftImage *bin, float max_thickness, float min_length)
{
  iftImage  *medial_axis  = iftMedialAxisTrans2D(bin,5.0,IFT_EXTERIOR);
  iftImage  *rec, *aux[3];
  iftAdjRel *A=iftCircular(sqrtf(2.0));

  /* Select (error) gaps whose maximum thickness is narrower than
     max_thickness */
 
  for (int p=0; p < medial_axis->n; p++) 
    if (medial_axis->val[p]>=(max_thickness*max_thickness))
      medial_axis->val[p]=0;

  /* Select (error) gaps whose minimum length is longer than
     min_length and eliminate terminal points of the medial axis */ 
  
  aux[0] = iftThreshold(medial_axis,1,IFT_INFINITY_INT,1);
  aux[1] = iftSelectCompAboveArea(aux[0], A, min_length);
  aux[2] = iftTerminalPoints2D(medial_axis);
  iftDestroyImage(&aux[0]);
  iftDestroyAdjRel(&A);
  iftSet *S=NULL;
  aux[0] = iftDilateBin(aux[2],&S,max_thickness);
  iftDestroySet(&S);
  iftDestroyImage(&aux[2]);

  for (int p=0; p < medial_axis->n; p++) 
    if ((aux[1]->val[p]==0)||(aux[0]->val[p]==1))
      medial_axis->val[p]=0;
  iftDestroyImage(&aux[0]);
  iftDestroyImage(&aux[1]);

  /* Reconstruct the gaps as object */
  
  aux[0] = iftShapeReconstruction(medial_axis, iftMaximumValue(bin));
  aux[1] = iftOr(aux[0],bin);
  rec    = iftCloseBasins(aux[1],NULL,NULL);
  iftDestroyImage(&medial_axis);
  iftDestroyImage(&aux[0]);
  iftDestroyImage(&aux[1]);

  return(rec);
}

int main(int argc, char *argv[]) 
{
  iftImage       *bin=NULL, *rec=NULL;
  timer          *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/


  if (argc!=4){
    printf("Usage: iftShapeRestoration2D <binary.pgm> <max_thickness> <min_length>\n");
    printf("Example 1: iftShapeRestoration2D fig1.pgm 5 20\n");
    exit(0);
  }

  // MEDIAL_AXIS

  bin    = iftReadImageByExt(argv[1]);

  t1     = iftTic(); 

  rec    = iftShapeRestoration(bin,atoi(argv[2]),atoi(argv[3]));

   t2     = iftToc(); 
   fprintf(stdout,"Restoration in %f ms\n",iftCompTime(t1,t2)); 


  iftWriteImageP2(rec,"result.pgm");
  
  iftDestroyImage(&bin);  
  iftDestroyImage(&rec);
 
  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}



