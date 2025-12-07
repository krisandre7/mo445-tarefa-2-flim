#include "ift.h"

void iftBinaryRegions(iftImage *label, iftAdjRel *A)
{
  iftImage *border = iftObjectBorders(label,A, false, true);
  int      nlabels, *area   = iftAllocIntArray((nlabels=iftMaximumValue(label))+1); 

  for (int p=0; p < label->n; p++) 
    area[label->val[p]]++;

  int imax=1;
  
  for (int i=2; i <= nlabels; i++) 
    if (area[i]>area[imax])
      imax = i;

  for (int p=0; p < label->n; p++){
    if (border->val[p]!=0) 
      label->val[p] = 0; 
    else {
      if (label->val[p]!=imax)
	label->val[p] = 255; 
      else
	label->val[p] = 0; 
    }
  }
  
  iftDestroyImage(&border);
} 



int main(int argc, char *argv[]) 
{
  iftImage       *orig, *marker, *label;
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

  if (argc!=4)
    iftError("Usage: iftSAM <normalized.scn (input)> <label.scn (output)> <h-value>","main");

  orig   = iftReadImageByExt(argv[1]);    
  
  t1     = iftTic();
  //marker = iftVolumeClose(orig,atoi(argv[3])); 
  marker = iftAddValue(orig,atoi(argv[3]));
  A      = iftSpheric(1.0);
  label  = iftWaterGray(orig,marker,A);
  iftBinaryRegions(label,A);
  iftDestroyAdjRel(&A);

  t2     = iftToc();
  fprintf(stdout,"Cells segmented in %f ms\n",iftCompTime(t1,t2));

  
  iftWriteImageByExt(label,argv[2]);

  iftDestroyImage(&orig);
  iftDestroyImage(&label);
  iftDestroyImage(&marker);
 
  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}

