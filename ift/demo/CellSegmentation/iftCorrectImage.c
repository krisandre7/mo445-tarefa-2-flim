#include "ift.h"

iftImage *iftCorrectImage(iftImage *orig, iftImage *fg, iftImage *dist, float K)
{
  iftImage *cimg;
  int       p,d,n,i;
  int       Dmax=(int)sqrt(iftMaximumValue(dist))+1, *value=NULL;
  iftSet    *S[Dmax],*T=NULL;
  int       maxval[Dmax], minval[Dmax], median[Dmax];

  /* Create one seed set for each iso-surface */

  for (d=0; d < Dmax; d++) 
    S[d]=NULL;

  n=0;
  for (p=0; p < orig->n; p++){
    if (fg->val[p]!=0){
      d = (int)sqrt(dist->val[p]);
      iftInsertSet(&S[d],p);
      n++;
    }
  }

  /* Compute the median and maximum values in each iso-surface */

  value = iftAllocIntArray(n);

  for (d=0; d < Dmax; d++){
    T = S[d]; i = 0; 
    maxval[d]=IFT_INFINITY_INT_NEG;
    minval[d]=IFT_INFINITY_INT;
    while (T!=NULL){
      p = T->elem;
      value[i] = orig->val[p]; 
      i++;
      T = T->next;
    }
    n = i; 
    median[d]=iftSelectTheKthElem(value,n,n/2);

    for (i=0; i < n; i++) {
      if (value[i]>maxval[d])
	maxval[d]=value[i];
      if (value[i]<minval[d])
	minval[d]=value[i];
    }
    printf("d %d maxval %d minval %d median %d \n",d,maxval[d],minval[d],median[d]);
    for (i=0; i < n; i++) 
      value[i]=0;
  }

  free(value);

  /* Correct image intensities with more emphasis as deeper is the
     iso-surface */

  cimg = iftCreateImage(orig->xsize,orig->ysize,orig->zsize);

  for (d=0; d < Dmax; d++){
    T = S[d];
    while (T!=NULL){
      p = T->elem;
      //      cimg->val[p] = (int)4095*pow((float)orig->val[p]/(float)maxval[d],K);
      cimg->val[p] = (int)(4095.0/(1 + exp(-K*(((float)orig->val[p]-(float)median[d])/((float)maxval[d]-(float)minval[d]+0.01)))));
      T = T->next;
    }
  }

  for (d=0; d < Dmax; d++) 
    iftDestroySet(&S[d]);

  iftCopyVoxelSize(orig,cimg);

  return(cimg);
}

int main(int argc, char *argv[]) 
{
  iftImage       *orig, *corr, *fg, *dist;
  timer          *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=6)
    iftError("Usage: iftCorrectImage <original.scn (input)> <foreground.scn (input)> <distance.scn (input)> <corrected.scn (output)> <factor K in (0,1]>","main");

  orig   = iftReadImageByExt(argv[1]);    
  fg     = iftReadImageByExt(argv[2]);    
  dist   = iftReadImageByExt(argv[3]);    
  
  t1     = iftTic();

  corr = iftCorrectImage(orig,fg,dist,atof(argv[5]));


  t2     = iftToc();
  fprintf(stdout,"Image corrected in %f ms\n",iftCompTime(t1,t2));

  iftWriteImageByExt(corr,argv[4]);

  iftDestroyImage(&corr);
  iftDestroyImage(&fg);  
  iftDestroyImage(&dist);
  iftDestroyImage(&orig);
 
  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}

