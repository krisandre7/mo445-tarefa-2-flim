#include "ift.h"

float iftCheckBorders(iftImage *img, iftImage *label, iftAdjRel *A)
{
  iftVoxel u,v;
  int i,p,q,br;
  int count,numBorders;
  count = 0;
  numBorders = 0;

  if ((img->xsize != label->xsize)||
      (img->ysize != label->ysize)||
      (img->zsize != label->zsize))
    iftError("Images must have the same domain","iftCheckBorders");

  for (p=0; p < img->n; p++) {
    u.x = iftGetXCoord(label,p);
    u.y = iftGetYCoord(label,p);
    u.z = iftGetZCoord(label,p);
    br = 0;
    for (i=0; i < A->n; i++) {
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if (iftValidVoxel(label,v)){
	q = iftGetVoxelIndex(label,v);
	if (label->val[p] < label->val[q]){
	  br++;
	  break;
	}
      }
    }
    if(img->val[p] == 255){
	  numBorders++;
	  if(br>0){
		  count++;
	  }
	}
  }
  printf("match: %d, Total: %d \n",count,numBorders);
  return ((float)count/(float)numBorders);
}

int iftLabelsWG(iftImage *basins, iftImage *pathval, iftAdjRel  *A)
{
  iftImage   *label=NULL;
  iftGQueue  *Q=NULL;
  int         i,p,q,l=1,tmp;
  iftVoxel    u,v;

  // Initialization

  label   = iftCreateImage(basins->xsize,basins->ysize,basins->zsize);
  Q       = iftCreateGQueue(iftMaximumValue(pathval)+2,pathval->n,pathval->val);

  for (p=0; p < basins->n; p++) {
    pathval->val[p] += 1;
    label->val[p]=NIL;
    iftInsertGQueue(&Q,p);
  }

  // Image Foresting Transform

  while(!iftEmptyGQueue(Q)) {
    p=iftRemoveGQueue(Q);

    if (label->val[p]==NIL) { // root voxel
      pathval->val[p]  -= 1;
      label->val[p]=l; l++;
    }

    basins->val[p] = pathval->val[p]; // set the reconstruction value

    u = iftGetVoxelCoord(basins,p);

    for (i=1; i < A->n; i++){
      v = iftGetAdjacentVoxel(A,u,i);
      if (iftValidVoxel(basins,v)){
	q = iftGetVoxelIndex(basins,v);
	if (Q->L.elem[q].color != BLACK){
	  tmp = MAX(pathval->val[p],basins->val[q]);
	  if (tmp < pathval->val[q]){
	    iftRemoveGQueueElem(Q,q);
	    label->val[q]      = label->val[p];
	    pathval->val[q]    = tmp;
	    iftInsertGQueue(&Q, q);
	  }
	}
      }
    }
  }

  iftDestroyGQueue(&Q);
  iftCopyVoxelSize(basins,label);

  return(l-1);
}

int main(int argc, char **argv) 
{
  iftImage       *img,   *orig, *imgGT,   *origGT;
  iftMImage      *input, *output;
  char            ext[10],*pos;
  timer          *t1=NULL,*t2=NULL;
  iftConvNetwork *convnet;
  int nroLbl;
  int vol;
  float perB;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/


  if (argc!=5)
    iftError("Usage: iftTestConvNetwork <image.[ppm,pgm]> <imageGT.[ppm,pgm]> <parameters.convnet> <VOLUME>","main");

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

  if (strcmp(ext,"ppm")==0){
   origGT   = iftReadImageP6(argv[2]);
  }else{
    if (strcmp(ext,"pgm")==0){
	  origGT   = iftReadImageP5(argv[2]);
    }else{
	  printf("Invalid image format: %s\n",ext);
	  exit(-1);
    }
  }


  pos = strrchr(argv[3],'.') + 1;
  sscanf(pos,"%s",ext);

  if (strcmp(ext,"convnet")==0){
    convnet = iftReadConvNetwork(argv[3]);
  }else{
    printf("Invalid file format: %s\n",ext);
    exit(-1);
  }

  vol = atoi(argv[4]);

  input  = iftImageToMImage(orig,YCbCr_CSPACE);
  t1     = iftTic();
  output = iftApplyConvNetwork(input,convnet);  
  t2     = iftToc();
  fprintf(stdout,"Convolution network executed in %f ms\n",iftCompTime(t1,t2));

  /* New code for John */

  convnet->with_weights = 1;
  iftWriteConvNetwork(convnet,"teste1.convnet");  

  char filename[200];

  for (int i=0; i < convnet->nkernels[0]; i++) {
    img = iftMImageToImage(output,255,i);
    sprintf(filename,"feat%d.pgm",i);
    iftWriteImageP5(img,filename);
    iftDestroyImage(&img);    
  }

  // for new Watergray
  iftAdjRel *A = iftCircular(sqrtf(2));
  img = iftMImageBasins(output, A);
  iftWriteImageP2(img,"basins.pgm");

  iftImage *marker = iftVolumeClose(img,vol);
  iftImage *label  = iftWaterGray(img,marker,A);
  nroLbl = iftLabelsWG(img,marker,A);
  iftColor RGB, YCbCr;

  RGB.val[0] = 255;
  RGB.val[1] = 0;
  RGB.val[2] = 0;
  YCbCr      = iftRGBtoYCbCr(RGB);
  iftAdjRel *B = iftCircular(0.0);

  int sz = (orig->xsize - img->xsize)/2;
  iftDestroyImage(&img);    
  imgGT        = iftRemFrame(origGT,sz);
  // Count number of matches
  perB = iftCheckBorders(imgGT,label,A);
  printf("Num labels result-new.ppm: %d, per: %f \n",nroLbl,perB);
  // Draw borders
  iftDrawBorders(imgGT,label,A,YCbCr,B);
  iftWriteImageP6(imgGT,"result-new.ppm");

  iftDestroyImage(&img);

  // for Traditional Watergray
  img = iftImageBasins(orig, A);
  iftDestroyImage(&marker);    
  iftDestroyImage(&label);

  marker = iftVolumeClose(img,vol);
  label  = iftWaterGray(img,marker,A);
  nroLbl = iftLabelsWG(img,marker,A);
  RGB.val[0] = 0;
  RGB.val[1] = 0;
  RGB.val[2] = 255;
  YCbCr      = iftRGBtoYCbCr(RGB);
  // Count number of matches
  perB = iftCheckBorders(origGT,label,A);
  printf("Num labels result-old.ppm: %d, per: %f \n",nroLbl,perB);
  // Draw borders
  iftDrawBorders(origGT,label,A,YCbCr,B);
  iftWriteImageP6(origGT,"result-old.ppm");



  iftDestroyImage(&img);    
  iftDestroyImage(&orig);    
  iftDestroyImage(&marker);    
  iftDestroyImage(&label);
  iftDestroyAdjRel(&A);
  iftDestroyAdjRel(&B);

  /* end of John's code */

  iftDestroyMImage(&input);
  iftDestroyMImage(&output);
  iftDestroyConvNetwork(&convnet);

 
  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   


  return(0);

}
