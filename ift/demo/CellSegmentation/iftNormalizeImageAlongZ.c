#include "ift.h"

iftImage *iftNormalizeImageAlongZ(iftImage *orig, int maxval)
{
  iftImage *cimg = iftCreateImage(orig->xsize,orig->ysize,orig->zsize);
  float    *Imax = iftAllocFloatArray(orig->zsize), *Imin=iftAllocFloatArray(orig->zsize);
  iftVoxel  u;
  int       p;

  for (u.z=0; u.z < orig->zsize; u.z++) {
    Imax[u.z]=IFT_INFINITY_FLT_NEG; Imin[u.z]=IFT_INFINITY_FLT;
    for (u.y=0; u.y < orig->ysize; u.y++) {
      for (u.x=0; u.x < orig->xsize; u.x++) {
	p = iftGetVoxelIndex(orig,u);
	if (orig->val[p] > Imax[u.z])
	  Imax[u.z] = orig->val[p];
	if (orig->val[p] < Imin[u.z])
	  Imin[u.z] = orig->val[p];
      }
    }
    //    printf("Imin[%d] = %f Imax[%d] = %f\n",u.z,Imin[u.z],u.z,Imax[u.z]); 
  }

  for (p=0; p < orig->n; p++){
    int z = iftGetZCoord(orig,p);
    cimg->val[p] = (int)(maxval*(orig->val[p]-Imin[z])/(Imax[z]-Imin[z]));
  }

  free(Imax);
  free(Imin);

  iftCopyVoxelSize(orig,cimg);

  return(cimg);
}

int main(int argc, char *argv[]) 
{
  iftImage       *orig, *norm;
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
    iftError("Usage: iftNormalizeImageAlongZ <original.scn (input)> <normalized.scn (output)> <maximum output value (e.g., 4095)>","main");

  orig   = iftReadImageByExt(argv[1]);    
  
  t1     = iftTic();

  norm = iftNormalizeImageAlongZ(orig,atoi(argv[3]));
  iftDestroyImage(&orig);

  t2     = iftToc();
  fprintf(stdout,"Normalized along Z in %f ms\n",iftCompTime(t1,t2));

  iftWriteImageByExt(norm,argv[2]);

  iftDestroyImage(&norm);
 
  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}

