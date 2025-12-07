#include "ift.h"

iftImage *iftShootDistanceMap(iftImage *fg)
{
  iftImage  *dist, *root;
  iftSet    *S=NULL;
  iftAdjRel *A=iftSpheric(1.0);
  
  for (int p=0; p < fg->n; p++) {
    if (fg->val[p]!=0){
      iftVoxel u = iftGetVoxelCoord(fg,p);
      if (u.z < fg->zsize){
	for (int i=1; i < A->n; i++){
	  iftVoxel v = iftGetAdjacentVoxel(A,u,i); 
	  if (iftValidVoxel(fg,v)){
	    int q = iftGetVoxelIndex(fg,v);
	    if (fg->val[p]!=fg->val[q])
	      iftInsertSet(&S,p);
	  }
	}
      }
    }
  }
  iftDestroyAdjRel(&A);
  A = iftSpheric(sqrtf(3.0));
  iftMultiLabelShellDistTransFromSet(S, fg, A, IFT_INTERIOR, IFT_INFINITY_DBL,
				     &dist, &root);

  iftDestroyAdjRel(&A);
  iftDestroyImage(&root);
  iftDestroySet(&S);

  return(dist);
}

int main(int argc, char *argv[]) 
{
  iftImage       *fg, *dist;
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
    iftError("Usage: iftShootDistanceMap <foreground.scn (input)> <distance.scn (output)> ","main");

  fg   = iftReadImageByExt(argv[1]);    
  
  t1     = iftTic();

  dist = iftShootDistanceMap(fg);

  t2     = iftToc();
  fprintf(stdout,"Shoot distance map computed in %f ms\n",iftCompTime(t1,t2));

  iftWriteImageByExt(dist,argv[2]);

  iftDestroyImage(&dist);
  iftDestroyImage(&fg);
 
  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}

