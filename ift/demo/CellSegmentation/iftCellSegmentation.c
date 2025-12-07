#include "ift.h"

void iftSelectEdgePoints2D(iftImage *basins,int p,float radius,int *p0,int *p1)
{
  iftVoxel u0, u1, center = iftGetVoxelCoord(basins,p);
  int      alpha, p0max, p1max;

  *p0  = *p1   = p;  
  u0.z =  u1.z = 0;
  for (alpha=0; alpha < 180; alpha++){
    u0.x = center.x + radius*cos(alpha*IFT_PI/180.0);
    u0.y = center.y + radius*sin(alpha*IFT_PI/180.0);
    u1.x = center.x + radius*cos((180+alpha)*IFT_PI/180.0);
    u1.y = center.y + radius*sin((180+alpha)*IFT_PI/180.0);
    if (iftValidVoxel(basins,u0)&&iftValidVoxel(basins,u1)){
      p0max = iftGetVoxelIndex(basins,u0);
      p1max = iftGetVoxelIndex(basins,u1);
      if ((basins->val[p0max]>basins->val[*p0])&&(basins->val[p1max]>basins->val[*p1])){
	*p0 = p0max;
	*p1 = p1max;
      }
    }
  }
}

iftSet *iftOptimumEdge(int p0, int p1, iftImage *basins, int maxval, iftGQueue **Q,iftImage *cost,iftImage *pred,int *S,int *nelems,iftAdjRel *A)
{
  int p,q,i,tmp;
  iftVoxel u, v;

  cost->val[p0] = 0; (*nelems)=0;
  iftInsertGQueue(Q,p0); S[0]=p0; (*nelems)++;    

  while (!iftEmptyGQueue(*Q)) {
    p = iftRemoveGQueue(*Q);
    u = iftGetVoxelCoord(basins, p);
    
    if (p == p1){
      break;
    }

    for (i = 1; i < A->n; i++){
      v = iftGetAdjacentVoxel(A, u, i);	
      if (iftValidVoxel(basins, v))
	{
	  q = iftGetVoxelIndex(basins, v);	    
	  if ((*Q)->L.elem[q].color != IFT_BLACK){
	    tmp = cost->val[p] + maxval - basins->val[q];
	    if (tmp < cost->val[q]){
	      if ((*Q)->L.elem[q].color == IFT_GRAY){
		iftRemoveGQueueElem(*Q,q);
	      }
	      cost->val[q]=tmp;
	      pred->val[q]=p;
	      iftInsertGQueue(Q,q);
	      S[*nelems]=q; (*nelems)++;
	    }
	  }
	}
    }
  }
  iftSet *E=NULL;
  q   = p1;
  while(q != IFT_NIL){
    iftInsertSet(&E,q); 
    q    = pred->val[q];
  }
  
  iftResetGQueueForVoxelList(*Q,S,*nelems);
  for (i=0; i < *nelems; i++) {
    cost->val[S[i]]=IFT_INFINITY_INT;
    pred->val[S[i]]=IFT_NIL;
  }

  return(E);
}

iftImage *iftEnhanceEdgesByLivewire(iftImage *basins, float radius)
{
  iftAdjRel *A;
  iftGQueue *Q;
  iftImage  *cost, *edge, *pred;
  int        maxval = iftMaximumValue(basins);
  int       *S, nelems=0;
  int        p0, p1;
    
  cost = iftCreateImage(basins->xsize,basins->ysize,basins->zsize);
  pred = iftCreateImage(basins->xsize,basins->ysize,basins->zsize);
  edge = iftCreateImage(basins->xsize,basins->ysize,basins->zsize);
  Q    = iftCreateGQueue(maxval+1,basins->n,cost->val);
  S    = iftAllocIntArray(basins->n);
    
  if (iftIs3DImage(basins))
    A = iftSpheric(1.0);
  else
    A = iftCircular(1.0);

  for (int p = 0; p < cost->n; p++){
    cost->val[p] = IFT_INFINITY_INT;
    pred->val[p] = IFT_NIL;
  }

  if (!iftIs3DImage(basins)){
    for (int p = 0; p < cost->n; p++){
      iftSelectEdgePoints2D(basins,p,radius,&p0,&p1);
      
      if ((p0 != p1)&&(basins->val[p0]>basins->val[p])&&(basins->val[p1]>basins->val[p])){
	iftSet *E=iftOptimumEdge(p0,p,basins,maxval,&Q,cost,pred,S,&nelems,A);
	int tmp=0;
	while(E!=NULL){
	  int q = iftRemoveSet(&E);
	  tmp = iftMax(tmp,basins->val[q]);
	}
	E = iftOptimumEdge(p,p1,basins,maxval,&Q,cost,pred,S,&nelems,A);
	while(E!=NULL){
	  int q = iftRemoveSet(&E);
	  tmp = iftMax(tmp,basins->val[q]);
	}
	edge->val[p] = tmp;
      }else{
	edge->val[p] = iftMax(edge->val[p],basins->val[p]);
      }
    }
  }else{
    printf("Not implemented yet\n");
  }
  
  iftDestroyGQueue(&Q);
  iftDestroyImage(&cost);
  iftFree(S);
  iftDestroyAdjRel(&A);

  return(edge);
}

iftImage *iftLeakingCell(iftImageForest *fst, iftImage *mask)
{
  iftImage *leaf   = iftLeafVoxels(fst->pred,fst->A);
  iftColor RGB, YCbCr;
  iftAdjRel *B;
  int      normval  = iftNormalizationValue(iftMaximumValue(fst->img));
  int      Lmax     = iftMaximumValue(fst->label);
  int      *nleaves = iftAllocIntArray(Lmax+1);
  iftImage *cell    = iftCopyImage(fst->img);
    
  
  for (int p=0; p < fst->img->n; p++) {    
    if (leaf->val[p] && (mask->val[p]==0)){
      nleaves[fst->label->val[p]]++;
    }
  }
  iftDestroyImage(&leaf);

  int Cmax=0;
  for(int i=1; i <= Lmax; i++)
    if (nleaves[i]>Cmax)
      Cmax= nleaves[i];
  
  for (int p=0; p < fst->img->n; p++) {    
    cell->val[p] = (int)(normval*(float)nleaves[fst->label->val[p]]/(float)Cmax);
  }
  iftSetCbCr(cell,normval);
	     
  for (int p=0; p < fst->img->n; p++) {    
    if (mask->val[p]==0){
      RGB.val[0] = 0;
      RGB.val[1] = cell->val[p];
      RGB.val[2] = cell->val[p];
      YCbCr      = iftRGBtoYCbCr(RGB,normval);
      cell->val[p] = fst->img->val[p];
      cell->Cb[p]  = YCbCr.val[1];
      cell->Cr[p]  = YCbCr.val[2];    
    }
  }


  if (iftIs3DImage(mask))
    B = iftSpheric(1.0);
  else
    B = iftCircular(1.0);
  
  RGB.val[0] = normval;
  RGB.val[1] = 0;
  RGB.val[2] = normval;
  YCbCr      = iftRGBtoYCbCr(RGB,normval);
  
  for (int p=0; p < fst->img->n; p++) {
    if (fst->root->val[p]==p){
      iftVoxel u = iftGetVoxelCoord(fst->img,p);
      iftDrawPoint(cell, u, YCbCr, B, normval);
    }
  }

  iftFree(nleaves);
  iftDestroyAdjRel(&B);  

  return(cell);
}

int main(int argc, char *argv[]) 
{
  iftImage        *img=NULL,*mask=NULL,*diff=NULL,*dil=NULL;
  iftImage        *marker=NULL, *allborders=NULL, *border=NULL;
  iftImageForest  *fst = NULL;
  iftAdjRel       *A=NULL,*B=NULL;
  iftSet          *S=NULL;
  timer           *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=3)
    iftError("Usage: iftCellSegmentation <image.[scn,pgm]> <height>","main");

  img   = iftReadImageByExt(argv[1]);    

  t1 = iftTic();

  if (iftIs3DImage(img)){
    A      = iftSpheric(1.0);
    B      = iftSpheric(0.0);
  }else{
    A      = iftCircular(1.0);
    B      = iftCircular(0.0);
  }
  
  fst                  = iftCreateImageForest(img, A);  
  marker               = iftAddValue(img,3);
  iftWaterGrayForest(fst,marker);
  allborders           = iftObjectBorders(fst->label,fst->A,false,false);
  iftDestroyImage(&marker);
  iftDestroyImageForest(&fst);

  fst                  = iftCreateImageForest(img, A);  
  marker               = iftAddValue(img,atoi(argv[2]));
  iftWaterGrayForest(fst,marker);
  border               = iftObjectBorders(fst->label,fst->A,false,false);

  mask                 = iftCreateImage(border->xsize,border->ysize,border->zsize);
  diff                  = iftCreateImage(border->xsize,border->ysize,border->zsize);
  S                    = NULL;
  dil                  = iftDilateBin(border,&S,1.0);
  iftDestroySet(&S);  
  for (int p=0; p < dil->n; p++){
    if (dil->val[p] && allborders->val[p])
      mask->val[p] = 255;
    else
      diff->val[p]  = allborders->val[p];
  }
  iftDestroyImage(&dil);
  
  iftImage *cell   = iftLeakingCell(fst,mask);

  t2     = iftToc(); 

  fprintf(stdout,"cell segmentation in %f ms with %d cells\n",iftCompTime(t1,t2),iftMaximumValue(fst->label));


  if (iftIs3DImage(img)){
    iftWriteImageByExt(border,"result.scn");
  }else{
    iftColor RGB, YCbCr;
    int  normval = iftNormalizationValue(iftMaximumValue(img));
    RGB.val[0] = normval;
    RGB.val[1] = normval;
    RGB.val[2] = 0;
    YCbCr      = iftRGBtoYCbCr(RGB,normval);
    iftDrawObject(cell, border, YCbCr, B);
    RGB.val[0] = 0;
    RGB.val[1] = normval;
    RGB.val[2] = 0;
    YCbCr      = iftRGBtoYCbCr(RGB,normval);
    iftDrawObject(cell, diff, YCbCr, B);
    iftWriteImageByExt(cell,"result.png");
  }

  iftDestroyImage(&mask);  
  iftDestroyImage(&cell);  
  iftDestroyAdjRel(&A);
  iftDestroyImage(&img);  
  iftDestroyImage(&marker);  
  iftDestroyImage(&border);  
  iftDestroyImage(&allborders);  
  iftDestroyImage(&diff);
  iftDestroyImageForest(&fst);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}

