#include "ift.h"

iftLabeledSet *iftBorderMarkers(iftImage *orig, iftImage *gt, int nseeds)
{
  iftLabeledSet *B=NULL;
  iftImage      *border=iftCreateImage(gt->xsize,gt->ysize,gt->zsize);
  iftImage      *non_border=iftCreateImage(gt->xsize,gt->ysize,gt->zsize);
  int            i,p,q;
  iftVoxel       u,v;
  int           *index = iftAllocIntArray(gt->n), nseeds1, nseeds2;
  iftImage      *basins, *marker, *label, *background; 
  iftAdjRel     *A;

  /* Compute background away from object */

  iftSet *S=NULL;
  iftImage *dil=iftDilateBin(gt,&S,10);
  iftDestroySet(&S);
  background = iftComplement(dil);
  iftDestroyImage(&dil);

  /* Compute image basins (constrast) and watershed by grayscale marker */
  
  A = iftCircular(sqrtf(2.0)); 
  basins = iftImageBasins(orig,A);
  marker = iftVolumeClose(basins,100);
  label  = iftWaterGray(basins,marker,A);
  iftDestroyImage(&marker); 
  iftDestroyAdjRel(&A);

  /* Mark voxels on object borders */

  A      = iftCircular(1.0);

  for (p=0; p < gt->n; p++){ 
    u = iftGetVoxelCoord(gt,p);
    for (i=1; i < A->n; i++) {
      v = iftGetAdjacentVoxel(A,u,i);
      if (iftValidVoxel(gt,v)){
	q = iftGetVoxelIndex(gt,v);
	if (gt->val[q]!=gt->val[p]){
	  border->val[p]=1;
	  break;
	}
      }
    }
  }
  
  /* Mark non-border voxels */

  for (p=0; p < label->n; p++){ 
    u = iftGetVoxelCoord(label,p);
    for (i=1; i < A->n; i++) {
      v = iftGetAdjacentVoxel(A,u,i);
      if (iftValidVoxel(label,v)){
	q = iftGetVoxelIndex(label,v);
	if ((label->val[q]!=label->val[p])&&
	    (!border->val[q])&&
	    (!border->val[p])&&
	    (background->val[p])) {
	    non_border->val[p]=1;
	    break;
	}
      }
    }
  }

  iftDestroyImage(&background);
  iftDestroyAdjRel(&A);
  iftDestroyImage(&label);

  /* Sort voxels by contrast */

  for (p=0; p < basins->n; p++) 
    index[p]=p;
  iftBucketSort(basins->val, index, basins->n, INCREASING);
  iftDestroyImage(&basins);

  /* Select seeds (markers) with border and non-border labels such
     that they are the most difficult voxels for classification by
     contrast */
 
  nseeds2=0;
  for (i=0; i < gt->n; i++) {
    p = index[i];
    if ((nseeds2 < nseeds/2) && border->val[p] ) {
      iftInsertLabeledSet(&B,p,1);
      nseeds2 ++;
    }
  }

  nseeds1=0;
  for (i=gt->n-1; i >= 0; i--) {
    p = index[i];
    if ((nseeds1 < nseeds/2) && non_border->val[p] ){ 
      iftInsertLabeledSet(&B,p,0);
      nseeds1 ++;
    }
  }
  
  free(index);
  iftDestroyImage(&border);
  iftDestroyImage(&non_border);

  return(B);
}

int main(int argc, char **argv) 
{
  iftImage        *orig, *gt;
  iftLabeledSet   *B=NULL;
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
    iftError("Usage: iftBorderMarkers <input.[ppm,pgm]> <gt.pgm> <number of markers> <markers.txt> ","main");

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
  B  = iftBorderMarkers(orig,gt,atoi(argv[3]));
  t2 = iftToc();
  fprintf(stdout,"Markers computed in %f ms\n",iftCompTime(t1,t2));

  iftWriteSeeds2D(argv[4],B,orig);

  iftDataSet *Z = iftImageSeedsToDataSet(orig,B);
  iftImage *img = iftDrawVoxelSamples(Z,CLASS,"iftImage");
  iftWriteImageP6(img,"markers.ppm");

  iftDestroyImage(&img);
  iftDestroyDataSet(&Z);
  iftDestroyLabeledSet(&B);
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
