#include "ift.h"

Image *myFastDilate(Image *I,Set **S, float radius)
{
  Image  *J=CreateImage(I->ncols,I->nrows);
  AdjRel *A=Circular(sqrt(2)),*A4=Circular(1.0);
  int    nspels,*V=AllocIntArray(nspels=I->ncols*I->nrows);
  int    *R=AllocIntArray(nspels);
  GQueue *Q=CreateGQueue(1024,nspels,V);
  int    p,q,i,tmp;
  float  r2=(radius*radius);
  Pixel  u,v,w;
  char   first_time=0;

  // Initialization

  if (*S==NULL)
    first_time=1;

  for(p=0; p < nspels; p++) {
    J->val[p]=I->val[p]; R[p]=p; 
    if (I->val[p]==0) // p is background
      V[p]=INT_MAX;
    else{ // p is object
      V[p]=INT_MIN;
      if (first_time){ // detect if p is border
	u.x = p % I->ncols;
	u.y = p / I->ncols;
	for (i=1; i < A4->n; i++) {
	  v.x = u.x + A4->dx[i];
	  v.y = u.y + A4->dy[i];
	  if (ValidPixel(I,v.x,v.y)){
	    q = v.x + I->tbrow[v.y];
	    if (I->val[q]==0){ // insert p in the border set S
	      InsertSet(S,p);
	      break;
	    }
	  }
	}
      }
    }
  }

  while (*S != NULL){
    p   = RemoveSet(S); 
    V[p]= 0; 
    InsertGQueue(&Q,p);
  }


  // Euclidean IFT dilation

  while(!EmptyGQueue(Q)){
    p = RemoveGQueue(Q);
    if (V[p] <= r2) {
      J->val[p]=1; 
      u.x = p % I->ncols;
      u.y = p / I->ncols;
      w.x = R[p] % I->ncols;
      w.y = R[p] / I->ncols;
      for (i=1; i < A->n; i++) {
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(I,v.x,v.y)){
	  q   = v.x + I->tbrow[v.y];
	  if (V[q]>V[p]){	    
	    tmp = (v.x-w.x)*(v.x-w.x)+(v.y-w.y)*(v.y-w.y);
	    if (tmp < V[q]){
	      if (V[q]!=INT_MAX){
		RemoveGQueueElem(Q, q);
	      }
	      V[q]=tmp; R[q]=R[p];
	      InsertGQueue(&Q,q);
	    }
	  }
	}
      }
    }else{
      InsertSet(S,p);
    }
  }

  free(V); free(R);
  DestroyGQueue(&Q);
  DestroyAdjRel(&A);
  DestroyAdjRel(&A4);

  return(J);
}

Image *myFastErode(Image *I,Set **S, float radius)
{
  Image  *J=CreateImage(I->ncols,I->nrows);
  AdjRel *A=Circular(sqrt(2)),*A4=Circular(1.0);
  int    nspels,*V=AllocIntArray(nspels=I->ncols*I->nrows);
  int    *R=AllocIntArray(nspels);
  GQueue *Q=CreateGQueue(1024,nspels,V);
  int    p,q,i,tmp;
  float  r2=(radius*radius);
  Pixel  u,v,w;
  char   first_time=0;

  // Initialization

  if (*S==NULL)
    first_time=1;

  for(p=0; p < nspels; p++) {
    J->val[p]=I->val[p]; R[p]=p; 
    if (I->val[p]==1) // p is object
      V[p]=INT_MAX;
    else{ // p is background
      V[p]=INT_MIN;
      if (first_time){ // detect if p is border
	u.x = p % I->ncols;
	u.y = p / I->ncols;
	for (i=1; i < A4->n; i++) {
	  v.x = u.x + A4->dx[i];
	  v.y = u.y + A4->dy[i];
	  if (ValidPixel(I,v.x,v.y)){
	    q = v.x + I->tbrow[v.y];
	    if (I->val[q]==1){ // insert p in the border set S
	      InsertSet(S,p);
	      break;
	    }
	  }
	}
      }
    }
  }

  while (*S != NULL){
    p   = RemoveSet(S); 
    V[p]= 0; 
    InsertGQueue(&Q,p);
  }


  // Euclidean IFT erosion

  while(!EmptyGQueue(Q)){
    p = RemoveGQueue(Q);
    if (V[p] <= r2) {
      J->val[p]=0; 
      u.x = p % I->ncols;
      u.y = p / I->ncols;
      w.x = R[p] % I->ncols;
      w.y = R[p] / I->ncols;
      for (i=1; i < A->n; i++) {
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(I,v.x,v.y)){
	  q   = v.x + I->tbrow[v.y];
	  if (V[q]>V[p]){	    
	    tmp = (v.x-w.x)*(v.x-w.x)+(v.y-w.y)*(v.y-w.y);
	    if (tmp < V[q]){
	      if (V[q]!=INT_MAX){
		RemoveGQueueElem(Q, q);
	      }
	      V[q]=tmp; J->val[q]=0; R[q]=R[p];
	      InsertGQueue(&Q,q);
	    }
	  }
	}
      }
    }else{
      InsertSet(S,p);
    }
  }

  free(V); free(R);
  DestroyGQueue(&Q);
  DestroyAdjRel(&A);
  DestroyAdjRel(&A4);
  return(J);
}

Image *myFastClose(Image *I, float radius)
{
  Image *J,*K;
  Set *S=NULL;
  J = myFastDilate(I,&S,radius);
  K = myFastErode(J,&S,radius);
  DestroyImage(&J);
  DestroySet(&S);
  return(K);
}

Image *myFastOpen(Image *I, float radius)
{
  Image *J,*K;
  Set *S=NULL;
  J = myFastErode(I,&S,radius);
  K = myFastDilate(J,&S,radius);
  DestroyImage(&J);
  DestroySet(&S);
  return(K);
}

Image *myFastCloseRec(Image *I, float radius)
{
  Image *J,*K;
  AdjRel *A4=Circular(1.0);
  J = myFastClose(I,radius);
  K = SupRec(I,J,A4);
  DestroyImage(&J);
  DestroyAdjRel(&A4);
  return(K);
}

Image *myFastOpenRec(Image *I, float radius)
{
  Image *J,*K;
  AdjRel *A4=Circular(1.0);
  J = myFastOpen(I,radius);
  K = InfRec(I,J,A4);
  DestroyImage(&J);
  DestroyAdjRel(&A4);
  return(K);
}

int main(int argc, char **argv)
{
  Image  *img[10];
  Set *S=NULL;
  AdjRel *A=NULL;

  /*------- -------------------------------------------------*/
#ifndef _WIN32
  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;
#endif

  /*--------------------------------------------------------*/
  
  if (argc != 2) {
    fprintf(stderr,"usage: binmorph <orig.pgm> \n");
    exit(-1);
  }



  img[0]= ReadImage(argv[1]);

  img[1] = Threshold(img[0],250,255);
  DestroyImage(&img[0]);
  img[0] = AddFrame(img[1],10,0);
  WriteImage(img[0],"shape.pgm");
  A = Circular(sqrt(2));
  img[2]= DistTrans(img[0],A,2);
  WriteImage(img[2],"distance.pgm");
  
  img[3]=myFastDilate(img[0],&S,10.0);
  WriteImage(img[3],"dilation.pgm");
  DestroySet(&S);
  
  img[4]=myFastErode(img[0],&S,10.0);
  WriteImage(img[4],"erosion.pgm");
  DestroySet(&S);
  
  img[5]=myFastClose(img[0],10.0);
  WriteImage(img[5],"close.pgm");
  
  img[6]= myFastOpen(img[0],10.0);
  WriteImage(img[6],"open.pgm");
  img[7]=myFastCloseRec(img[0],10.0);
  WriteImage(img[7],"closerec.pgm");
  img[8]=myFastOpenRec(img[0],10.0);
  WriteImage(img[8],"openrec.pgm");
  
  DestroyImage(&img[0]);
  DestroyImage(&img[1]);
  DestroyImage(&img[2]);
  
  DestroyImage(&img[3]);
  
  DestroyImage(&img[4]);
  
  DestroyImage(&img[5]);
  
  DestroyImage(&img[6]);
  DestroyImage(&img[7]);
  DestroyImage(&img[8]);
  
  DestroyAdjRel(&A);

  /* ---------------------------------------------------------- */

#ifndef _WIN32
  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   
#endif

  return(0);
}
