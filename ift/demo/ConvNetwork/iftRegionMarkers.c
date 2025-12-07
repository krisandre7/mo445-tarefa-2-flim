#include "ift.h"

iftLabeledSet *iftRegionMarkers(iftImage *gt, int nseeds)
{
  iftLabeledSet *R=NULL;
  iftAdjRel *A;
  iftImage *dist;
  char *selected=iftAllocCharArray(gt->n);
  int p, obj_samples=0, bkg_samples=0;

  if (iftIs3DImage(gt)) 
    A = iftSpheric(sqrtf(3.0));
  else
    A = iftCircular(sqrtf(2.0));

  dist = iftEuclDistTrans(gt, A, IFT_BOTH, NULL, NULL, NULL);

  while ((obj_samples+bkg_samples) < nseeds) {
    p = iftRandomInteger(0,gt->n-1);
    if ((selected[p]==0)&&(dist->val[p] >= 100)){
      if (gt->val[p]!=0){
	if (obj_samples < nseeds/2) {
	  iftInsertLabeledSet(&R,p,1);
	  obj_samples++;
	}
      }else{
	if (bkg_samples < nseeds/2) {
	  iftInsertLabeledSet(&R,p,0);
	  bkg_samples++;
	}
      }
    }
  }
  free(selected);
  iftDestroyAdjRel(&A);
  iftDestroyImage(&dist);
  return(R);
}

int main(int argc, char **argv) 
{
  iftImage        *orig, *gt;
  iftLabeledSet   *R=NULL;
  char            ext[10],*pos;
  timer          *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=5)
    iftError("Usage: iftRegionMarkers <input.[ppm,pgm]> <gt.pgm> <number of markers> <markers.txt> ","main");

  pos = strrchr(argv[1],'.') + 1;
  sscanf(pos,"%s",ext);
  
  if (strcmp(ext,"ppm")==0){
    orig   = iftReadImageP6(argv[1]);    
  }else{
    if (strcmp(ext,"pgm")==0){
      orig   = iftReadImageP5(argv[1]);    
    }else{
      printf("Invalid image format: %s\n",ext);
      exit(-1);
    }
  }

  pos = strrchr(argv[2],'.') + 1;
  sscanf(pos,"%s",ext);
  
  if (strcmp(ext,"pgm")==0){
    gt  = iftReadImageP5(argv[2]);    
  }else{
    printf("Invalid image format: %s\n",ext);
    exit(-1);
  }


  /* Compute seeds with border and non-border labels */

  t1 = iftTic();
  R  = iftRegionMarkers(gt,atoi(argv[3]));
  t2 = iftToc();
  fprintf(stdout,"Markers computed in %f ms\n",iftCompTime(t1,t2));

  iftWriteSeeds2D(argv[4],R,orig);

  iftDataSet *Z = iftImageSeedsToDataSet(orig,R);
  iftImage *img = iftDrawVoxelSamples(Z,CLASS,"iftImage");
  iftWriteImageP6(img,"markers.ppm");

  iftDestroyImage(&img);
  iftDestroyDataSet(&Z);
  iftDestroyLabeledSet(&R);
  iftDestroyImage(&orig);
  iftDestroyImage(&gt);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   


  return(0);
}
