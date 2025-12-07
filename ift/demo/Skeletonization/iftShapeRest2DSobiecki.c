#include "ift.h"

void iftDrawWhiteDisk(iftImage *bin, int p, float radius)
{
  int        q,i,Imax=iftMaximumValue(bin);
  iftVoxel   u = iftGetVoxelCoord(bin,p), v;
  iftAdjRel *B = iftCircular(radius);

  for (i=0; i < B->n; i++) {
    v = iftGetAdjacentVoxel(B,u,i);
    if (iftValidVoxel(bin,v)){
      q = iftGetVoxelIndex(bin,v);
      bin->val[q]=Imax;
    }
  }
  iftDestroyAdjRel(&B);
}


iftImage *iftShapeRestoration(iftImage *bin, float radius)
{
  iftImage  *openclose = iftAsfOCBin(bin,radius);
  iftImage  *closeopen = iftAsfCOBin(bin,radius);
  iftAdjRel *A         = iftCircular(sqrtf(2.0));
  iftFImage *msskel    = iftMSSkel2D(openclose, A, IFT_INTERIOR);
  iftImage  *skel      = iftFThreshold(msskel,5.0,100.0,255);
  iftImage  *dist      = iftEuclDistTrans(closeopen, A, IFT_INTERIOR, NULL, NULL, NULL);
  iftSet    *F         = NULL;
  iftSet    *H         = NULL;
  iftImage  *rec       = iftCopyImage(bin);

  for (int p=0; p < bin->n; p++) {
    if (skel->val[p]!=0){ // this implies that error gaps must
			  // intersect the skeleton.
      if (bin->val[p]==0){
	iftInsertSet(&F,p);
      } else {
	iftInsertSet(&H,p);
      }
    }
  }

  while (F != NULL) {
    int p = iftRemoveSet(&F);
    iftSet *S = H; // F - it is interesting, but eliminate more
		   // details as well.
    int q, closest=IFT_NIL, d, mindist=IFT_INFINITY_INT;
    iftVoxel u = iftGetVoxelCoord(bin,p);

    while (S != NULL){
      q = S->elem;
      iftVoxel v = iftGetVoxelCoord(bin,q);
      d = iftPointDistance(u,v);
      if (d < mindist) {
	mindist = d;
	closest = q;
      }
      S = S->next;
    }
    
    iftDrawWhiteDisk(rec, closest, sqrtf(dist->val[closest]));
  }

  iftDestroyImage(&openclose);
  iftDestroyImage(&closeopen);
  iftDestroyAdjRel(&A);
  iftDestroyFImage(&msskel);
  iftDestroyImage(&skel);
  iftDestroyImage(&dist);
  iftDestroySet(&H);

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


  if (argc!=3)
    iftError("Usage: iftShapeRestoration2D <binary.pgm> <radius>","main");

  // MEDIAL_AXIS

  bin    = iftReadImageByExt(argv[1]);

  t1     = iftTic(); 

  rec    = iftShapeRestoration(bin,atof(argv[2]));

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



