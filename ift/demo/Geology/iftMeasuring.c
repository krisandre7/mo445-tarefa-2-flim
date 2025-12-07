#include "ift.h"

 void iftPathPropagation1(iftImage *weight, iftFImage *cost, iftImage *pred, iftImage *root)
{
  iftFHeap  *Q=NULL;
  int        i, p, q;
  float      tmp;
  iftVoxel   u, v;
  iftAdjRel *A=iftSpheric(sqrtf(3.0));

  iftMaximumValue(weight);

  /* Set all voxels at the first plane as seeds for geodesic path
     propagation */

  Q = iftCreateFHeap(weight->n, cost->val);

  for (p=0; p < weight->n; p++) {
    cost->val[p]=IFT_INFINITY_FLT;
    if (iftGetZCoord(weight,p)==0){
      cost->val[p]=0.0;
      pred->val[p]=IFT_NIL;
      root->val[p]=p;
      iftInsertFHeap(Q,p);
    }
  }

  /* Geodesic path propagation by Image Foresting Transform */

  while (!iftEmptyFHeap(Q))
  {
    p = iftRemoveFHeap(Q);
    u = iftGetVoxelCoord(weight, p);

    for (i = 1; i < A->n; i++)
    {
      v = iftGetAdjacentVoxel(A, u, i);
      if (iftValidVoxel(weight, v))
      {
        q   = iftGetVoxelIndex(weight, v);
	if(Q->color[q] != IFT_BLACK){
	  tmp = iftMax(cost->val[p],weight->val[q]);
	  if (tmp < cost->val[q]){
	    cost->val[q] = tmp;
	    pred->val[q] = p;
	    root->val[q] = root->val[p];
	    if(Q->color[q] == IFT_WHITE)
	      iftInsertFHeap(Q, q);
	    else
	      iftGoUpFHeap(Q, Q->pos[q]);
	  }
	}
      }
    }
  }
  iftDestroyFHeap(&Q);
  iftDestroyAdjRel(&A);
}


float iftFCham3(iftFImage *scn, int p, int q)
{
  iftVoxel u,v;
  float cost;
  int xysize = scn->xsize*scn->ysize;
  
  u.x = (p % xysize) % scn->xsize;
  u.y = (p % xysize) / scn->xsize;
  u.z = p / xysize;
  v.x = (q % xysize) % scn->xsize;
  v.y = (q % xysize) / scn->xsize;
  v.z = q / xysize;
  
     
  if ((((u.x == v.x)||(u.y == v.y))&&(u.z == v.z))||(((u.x == v.x)&&(u.y == v.y))&&(u.z != v.z)))
    {
			cost = scn->val[p]+5;
				return(cost);
		}
 	else
			{
			if ((u.x != v.x)&&(u.y != v.y)&&(u.z != v.z))
				{
					cost = scn->val[p]+9;
				return(cost);
				}
			else
			 	{
					cost = scn->val[p]+7;
				return(cost);
				}
			}
	
}

 void iftPathPropagation2(iftImage *weight, iftFImage *cost, iftImage *pred, iftImage *root)
{
  iftFHeap  *Q=NULL;
  int        i, p, q;
  //float      tmp;
  iftVoxel   u, v;
  iftAdjRel *A=iftSpheric(sqrtf(3.0));
  int count=0;
  int costval;
  //int otsu;


  iftMaximumValue(weight);
  //otsu = iftOtsu(weight);

  /* Set all voxels at the first plane as seeds for geodesic path
     propagation */

  Q = iftCreateFHeap(weight->n, cost->val);

  for (p=0; p < weight->n; p++) {
    if (weight->val[p]==0)
    	cost->val[p]=0;
    else
			cost->val[p]=IFT_INFINITY_FLT;
			
    if ((iftGetZCoord(weight,p)==0)&&(weight->val[p] == 1)){
      cost->val[p]=0.0;
      pred->val[p]=IFT_NIL;
      root->val[p]=p;
      iftInsertFHeap(Q,p);
      count++;
    }
  }
  
  /* Geodesic path propagation by Image Foresting Transform */

  while (!iftEmptyFHeap(Q))
  {
    p = iftRemoveFHeap(Q);
    u = iftGetVoxelCoord(weight, p);

    for (i = 1; i < A->n; i++)
    {
      v = iftGetAdjacentVoxel(A, u, i);
      if (iftValidVoxel(weight, v))
      {
        q   = iftGetVoxelIndex(weight, v);
	if(Q->color[q] != IFT_BLACK){
		costval =  iftFCham3(cost,p,q);
	  if (costval < cost->val[q]){
	    if(Q->color[q] == IFT_WHITE)
	      iftInsertFHeap(Q, q);
	    else
	      iftGoUpFHeap(Q, Q->pos[q]);
	    cost->val[q] = costval;
	    pred->val[q] = p;
	    root->val[q] = root->val[p];
	  }
	}
      }
    }
  }
  iftDestroyFHeap(&Q);
  iftDestroyAdjRel(&A);
}


float imf_ElZero(iftImage *weight, int p, int q)
{
	iftVoxel a, b;
	iftVoxel uo, uf;
	int i=0, j, f=0;
	float l0;
	iftImage *roi = NULL;
	iftFImage      *cost=NULL; 
  iftImage       *root=NULL, *pred=NULL;
	
	a = iftGetVoxelCoord(weight, p);	// fundo da imagem
	b = iftGetVoxelCoord(weight, q);	// topo da imagem
	
	uo.x	= iftMin(a.x, b.x);
	uo.y	= iftMin(a.y, b.y);
	uo.z	= iftMin(a.z, b.z);
	uf.x	= iftMax(a.x, b.x);
	uf.y	= iftMax(a.y, b.y);
	uf.z	= iftMax(a.z, b.z);
	
	//fprintf(stderr,"p:%d q:%d uo:%d %d %d uf:%d %d %d\n", p, q, uo.x, uo.y, uo.z, uf.x, uf.y, uf.z);

	iftBoundingBox bb = {.begin = uo, .end = uf};	
	roi = iftExtractROI (weight, bb);
	
	
	for ( j = 0; j < roi->n; j++)
	{
		if (iftGetZCoord(roi,j)==0) {
			if (roi->val[j]!=0) i = j;
		}
		if (iftGetZCoord(roi,j)==roi->zsize-1) { 
			if (roi->val[j]!=0) f = j;
		}
		roi->val[j] = 2;
	}
	
	roi->val[i] = 1;
	//iftWriteImage(roi,"check.scn");
	
	cost   	= iftCreateFImage(roi->xsize,roi->ysize,roi->zsize);
	root   	= iftCreateImage(roi->xsize,roi->ysize,roi->zsize);
	pred   	= iftCreateImage(roi->xsize,roi->ysize,roi->zsize);
	iftPathPropagation2(roi,cost,pred,root);
	
	
	//fprintf(stderr,"f:%d L:%f\n", f, cost->val[f]);
	
	l0 = cost->val[f];
	
	
	
	
	iftDestroyImage(&roi);
	iftDestroyImage(&root);
	iftDestroyImage(&pred);
	iftDestroyFImage(&cost);
	
return(l0);
}

int main(int argc, char *argv[]) 
{
  iftImage	 *weight=NULL, *input=NULL;
  iftFImage      *cost=NULL; 
  iftImage       *root=NULL, *pred=NULL;
  int maxval;
  float Fmaxval;
  int p;
  FILE *fp = NULL;
  timer          *t1=NULL,*t2=NULL;
	
  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=2)
    iftError("Usage: iftMeasuring <path_bin.scn>","main");
  
  t1 = iftTic();
  
  
  input 	= iftReadImage(argv[1]);
  maxval 	= iftMaximumValue(input);
  weight 	= iftThreshold(input, 1, maxval, 1);
  iftDestroyImage(&input);
  cost   	= iftCreateFImage(weight->xsize,weight->ysize,weight->zsize);
  pred   	= iftCreateImage(weight->xsize,weight->ysize,weight->zsize);
  root   	= iftCreateImage(weight->xsize,weight->ysize,weight->zsize);
  iftPathPropagation2(weight,cost,pred,root);
	
	iftWriteFImage(cost,"Fcost.scn");

//if (iftGetZCoord(weight,p)==0)

  t2     = iftToc(); 
  iftImage *aux = iftFImageToImage(cost,4095);
  iftWriteImage(aux,"teste.scn");
  iftDestroyImage(&aux);
  
  Fmaxval = iftFMaximumValue(cost);
  //fprintf(stderr,"Fmaxval:%f\n",Fmaxval);

	
	fp = fopen("measures.txt","w");
	for (p = cost->n; p > cost->n-(cost->xsize*cost->ysize); p--)	{
		if (cost->val[p] != 0) 
			{
			//fprintf(stderr,"%d:%f\t", p, cost->val[p]);
			//fprintf(stderr,"%d:%d\t", p, root->val[p]);
			//fprintf(stderr,"%d:%d\n",	root->val[p], root->val[root->val[p]]);
			//l0 = imf_ElZero(weight, p, root->val[p]);
			fprintf(fp,"%.0f %.0f %d\n", cost->val[p], imf_ElZero(weight, p, root->val[p]), 5*weight->zsize);
			}
			//fprintf(fp,"%f %f %f\n", cost->val[p], l0, l);
	}
	fclose(fp);
/*
	FILE *fp = fopen("tmp.txt","w");
	for (p = 0; p < pred->n; p++)	{
		if (iftGetZCoord(pred,p)==0) {
		
	fprintf(fp,"%d %d %d\n", p, root->val[p],	pred->val[p]);
	
		}
	}
fclose(fp);
*/

/*
p = 188432089;

	while ( p != root->val[p]){
		fprintf(stderr,"%d %d\n", p, pred->val[p]);
		p = pred->val[p];
		
	}
*/

	//imf_El(weight, 188432089, root->val[188432089]);


  fprintf(stdout,"Tortuosity computed in %f ms \n",iftCompTime(t1,t2));

  iftDestroyImage(&weight);
  iftDestroyFImage(&cost);
  iftDestroyImage(&pred);
  iftDestroyImage(&root);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   


  return(0);
}
