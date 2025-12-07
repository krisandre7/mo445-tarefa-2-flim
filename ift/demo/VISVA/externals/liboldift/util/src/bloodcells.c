#include "ift.h"

Image *DualWaterGrayMask(Image *img, int H, Image *mask, AdjRel *A)
{
  Image *cost=NULL,*label=NULL;
  GQueue *Q=NULL;
  int i,p,q,tmp,n,r=1;
  Pixel u,v;

  n     = img->ncols*img->nrows;
  cost  = CreateImage(img->ncols,img->nrows);
  label = CreateImage(img->ncols,img->nrows);
  Q     = CreateGQueue(MaximumValue(img)+1,n,cost->val);
  SetRemovalPolicy(Q,MAXVALUE);

  for (p=0; p < n; p++) {
    if (mask->val[p]){
      cost->val[p]=MAX(img->val[p]-H,0);
      InsertGQueue(&Q,p);
    }
  }

  while(!EmptyGQueue(Q)) {
    p=RemoveGQueue(Q);
    if (label->val[p]==0) {
      cost->val[p]=img->val[p];
      label->val[p]=r;
      r++;
    }
    u.x = p%img->ncols;
    u.y = p/img->ncols;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(img,v.x,v.y)){
	q = v.x + img->tbrow[v.y];
	if (mask->val[p]){
	  if (cost->val[q] < cost->val[p]){
	    tmp = MIN(cost->val[p],img->val[q]);
	    if (tmp > cost->val[q]){
	      UpdateGQueue(&Q,q,tmp);
	      label->val[q] = label->val[p];
	    }
	  }
	}
      }
    }
  }
  DestroyGQueue(&Q);
  DestroyImage(&cost);

  return(label);
}

Image *TreatOutliers(Image *label)
{
  Image *flabel=ift_CopyImage(label);
  int p,q,r,s,i,n=label->ncols*label->nrows;
  Pixel u,v;
  AdjRel *A=Circular(1.0);
  int *FIFO,first=0,last=0;
  int *color;
  Set *S = NULL;

  FIFO  = AllocIntArray(n);
  color = AllocIntArray(n);

  for(p=0; p < n; p++) {
    if (flabel->val[p]==0){
      FIFO[last]=p;
      InsertSet(&S, p);
      last++;
      while(first != last){
	q = FIFO[first];
	color[q] = BLACK;
	u.x = q % label->ncols;
	u.y = q / label->ncols;
	first++;
	for (i=1; i < A->n; i++) {
	  v.x = u.x + A->dx[i];
	  v.y = u.y + A->dy[i];
	  r = v.x + label->tbrow[v.y];
	  if (ValidPixel(label,v.x,v.y)){
	    if (color[r] == WHITE) {
	      if (label->val[r]==0){
		FIFO[last] = r;
		color[r] = GRAY;
		InsertSet(&S, r);
		last++;
	      }else{
		flabel->val[p] = label->val[r];
		first=last=0;
		break;
	      }
	    }
	  }
	}
      }
      while (S != NULL) {
	s = RemoveSet(&S);
	color[s] = WHITE;
      }
    }
  }
  DestroyAdjRel(&A);
  free(FIFO);
  free(color);

  return(flabel);
}

int GetMinValueAround(Image *img, Pixel u, AdjRel *A)
{
  Pixel v;
  int i,q,min=img->val[u.x+img->tbrow[u.y]];

  
  for (i=1; i < A->n; i++) {
    v.x = u.x + A->dx[i];
    v.y = u.y + A->dy[i];
    if (ValidPixel(img,v.x,v.y)){
      if (img->val[q=v.x+img->tbrow[v.y]]<min)
	min = img->val[q];
    }
  }
  
  return(min);
}

Image *MyRemDomes(Image *img)
{
  AdjRel *A=NULL,*B=Circular(7.0);
  Image *marker=NULL,*oimg=NULL;
  int Imin,i,j,x,y;
  Pixel u;
  
  Imin  = MinimumValue(img);
  marker  = CreateImage(img->ncols,img->nrows);
  SetImage(marker,MAX(Imin-1,0));
  for (y=0; y < marker->nrows; y++) {
    i = marker->tbrow[y]; j = marker->ncols-1+marker->tbrow[y];
    u.x=0; u.y=y;
    marker->val[i] = GetMinValueAround(img,u,B);
    u.x=marker->ncols-1; u.y=y;
    marker->val[j] = GetMinValueAround(img,u,B);
  }
  for (x=0; x < marker->ncols; x++) {
    i = x+marker->tbrow[0]; j = x+marker->tbrow[marker->nrows-1]; 
    u.x=x; u.y=0;
    marker->val[i] = GetMinValueAround(img,u,B);
    u.x=x; u.y=marker->nrows-1;
    marker->val[j] = GetMinValueAround(img,u,B);
  }
  DestroyAdjRel(&B);
  A     = Circular(1.0);
  oimg  = InfRec(img,marker,A);
  
  DestroyImage(&marker);
  DestroyAdjRel(&A);

  return(oimg);
}


int main(int argc, char **argv)
{
  timer    *t1=NULL,*t2=NULL;
  Image    *img=NULL,*label=NULL,*odomes=NULL,*bin=NULL;
  Image    *marker=NULL, *dist=NULL;
  CImage   *cimg=NULL;
  Features *feat=NULL;
  Subgraph *sg=NULL;
  int p,n,q,i,imin;
  float *mean=NULL,*nelems=NULL,Imin;
  Pixel u,v;
  AdjRel *A=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);
  struct mallinfo info;
  int MemDinInicial, MemDinFinal;
  free(trash);
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  // use boodcells.pgm

  if (argc!=2){
    fprintf(stderr,"Usage: bloodcells <image.pgm> \n");
    fprintf(stderr,"image.pgm: image to be classified\n");
    exit(-1);
  }

  char *ext = strrchr(argv[1],'.');

  if(strcmp(ext,".pgm")==0){
    img    = ReadImage(argv[1]);    
    n      = img->ncols*img->nrows;
    A = Circular(5.0);
    marker = Open(img,A);
    WriteImage(marker,"marker.pgm");
    odomes = InfRec(img,marker,A);
    DestroyImage(&marker);
    WriteImage(odomes,"odomes.pgm");
    DestroyAdjRel(&A);
    //    odomes = MyRemDomes(img); // remove nucleos
    sg     = RandomSampl(odomes,200); 
    feat   = LMSImageFeats(odomes,3);
  }else{
    printf("Invalid image format\n");
    exit(1);
  }


  t1 = Tic();


  // Compute clusters 

  SetSubgraphFeatures(sg,feat);
  opf_BestkMinCut(sg,(int)(0.1*sg->nnodes),(int)(0.4*sg->nnodes));
  opf_OPFClustering(sg);
  bin=ImageClassKnnGraph(sg,feat); 
  label = TreatOutliers(bin);
  WriteImage(label,"clusters.pgm");
  DestroyImage(&bin);

  // Select the darkest one

  mean   = AllocFloatArray(sg->nlabels);
  nelems = AllocFloatArray(sg->nlabels);
  for (p=0; p < n; p++) {
    mean[label->val[p]-1] += odomes->val[p];
    nelems[label->val[p]-1] += 1;
  }
  for (i=0; i < sg->nlabels; i++)
    mean[i]/=nelems[i];
  imin=0; Imin=mean[0];
  for (i=1; i < sg->nlabels; i++)
    if (mean[i]<Imin){
      Imin = mean[i];
      imin = i;
    }
  bin=Threshold(label,imin+1,imin+1);
  free(mean); free(nelems);
  WriteImage(bin,"bin.pgm");

  // Compute the Distance Transform

  A = Circular(1.5);
  dist=DistTrans(bin,A,0);
  WriteImage(dist,"dist.pgm");
  DestroyImage(&label);
  label  = DualWaterGrayMask(dist,15,bin,A);

  t2 = Toc();

  fprintf(stdout,"bloodcells time in %f ms\n",CTime(t1,t2));

  // Draw samples used for clustering

  cimg = CreateCImage(img->ncols,img->nrows);
  for (p=0; p < n; p++){
    cimg->C[0]->val[p]= cimg->C[1]->val[p]= cimg->C[2]->val[p]=odomes->val[p];
  }
  for (i=0; i < sg->nnodes; i++) 
    DrawCPoint(cimg,sg->node[i].pixel%img->ncols,sg->node[i].pixel/img->ncols,3.0,255,0,0);   WriteCImage(cimg,"samples.ppm");
  DestroyCImage(&cimg);


  // Draw final result

  DestroyAdjRel(&A);
  A    = Circular(1.0);
  cimg = CreateCImage(img->ncols,img->nrows);
  for (p=0; p < n; p++){
    u.x = p%label->ncols;
    u.y = p/label->ncols;
    cimg->C[0]->val[p]= cimg->C[1]->val[p]= cimg->C[2]->val[p]=img->val[p];
    for (i=0; i < A->n; i++) {
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(label,v.x,v.y)){
	q = v.x + label->tbrow[v.y];
	if (label->val[q]!=label->val[p]){
	  cimg->C[0]->val[p]=255;
	  cimg->C[1]->val[p]=255;
	  cimg->C[2]->val[p]=0;
	  break;
	}
      }
    }
  }
  DestroyAdjRel(&A);

  WriteCImage(cimg,"result.ppm");

  
  DestroyImage(&dist);
  DestroyImage(&img);
  DestroyImage(&bin);
  DestroyImage(&odomes);
  DestroyImage(&label);
  DestroyCImage(&cimg);
  DestroySubgraph(&sg);
  DestroyFeatures(&feat);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);

  return(0);
}



