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


float iftFCham3D_Geo(iftFImage *scn, int p, int q)
{
  iftVoxel u,v;
  float cost;

  
  u = iftFGetVoxelCoord(scn, p);
  v = iftFGetVoxelCoord(scn, q);


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



float iftFCham3D_Hyd(iftFImage *scn, iftImage *weight, int p, int q)
{
  iftVoxel u,v;
  float cost;
   
  u = iftFGetVoxelCoord(scn, p);
  v = iftFGetVoxelCoord(scn, q);

     
  if ((((u.x == v.x)||(u.y == v.y))&&(u.z == v.z))||(((u.x == v.x)&&(u.y == v.y))&&(u.z != v.z)))
    {
			cost = scn->val[p]+5*(float)weight->val[q];
				return(cost);
		}
 	else
			{
			if ((u.x != v.x)&&(u.y != v.y)&&(u.z != v.z))
				{
					cost = scn->val[p]+9*(float)weight->val[q];
				return(cost);
				}
			else
			 	{
					cost = scn->val[p]+7*(float)weight->val[q];
				return(cost);
				}
			}
	
}


void iftPathPropagationVolume(iftImage *weight, iftFImage *cost, iftImage *pred, iftImage *root)
{
  iftFHeap  *Q=NULL;
  int        i, p, q;
  iftVoxel   u, v;
  iftAdjRel *A=iftSpheric(sqrtf(3.0));
  float costval;
  
  iftMaximumValue(weight); 

  Q = iftCreateFHeap(weight->n, cost->val);

  for (p=0; p < weight->n; p++) {
    cost->val[p] = IFT_INFINITY_FLT;
		if ((iftGetZCoord(weight,p)==0)){
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
					costval =  iftFCham3D_Hyd(cost,weight,p,q);
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
  
	float costmaxval = iftFMaximumValue(cost);
  printf("Fcost->max:%f\n",costmaxval);
  
  iftDestroyFHeap(&Q);
  iftDestroyAdjRel(&A);
}

void iftPathPropagationVolumeOnMask(iftImage *weight, iftFImage *cost, iftImage *pred, iftImage *root, iftImage *mask)
{
  iftFHeap  *Q=NULL;
  int        i, p, q;
  iftVoxel   u, v;
  iftAdjRel *A=iftSpheric(sqrtf(3.0));
  float costval;
  
	iftMaximumValue(weight); 

  Q = iftCreateFHeap(weight->n, cost->val);

  for (p=0; p < weight->n; p++) {
    cost->val[p] = IFT_INFINITY_FLT;
    pred->val[p] = IFT_NIL;
    root->val[p] = p;
    if ((iftGetZCoord(weight,p)==0) && mask->val[p] > 0){
      cost->val[p]=0.0;    
      iftInsertFHeap(Q,p);
    }
  }  
  
  /* Geodesic path propagation by Image Foresting Transform */

  while (!iftEmptyFHeap(Q)) {
    p = iftRemoveFHeap(Q);
    u = iftGetVoxelCoord(weight, p);
    
    for (i = 1; i < A->n; i++)  {
      v = iftGetAdjacentVoxel(A, u, i);
      if (iftValidVoxel(weight, v)) {
	q   = iftGetVoxelIndex(weight, v);
	if(Q->color[q] != IFT_BLACK && mask->val[q] > 0) {
	  costval =  iftFCham3D_Hyd(cost,weight,p,q);
	  
	  if (costval < cost->val[q]) {
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
  
  float costmaxval = iftFMaximumValue(cost);
  printf("Fcost->max:%f\n",costmaxval);
  
  iftDestroyFHeap(&Q);
  iftDestroyAdjRel(&A);
}


void iftHydroPathPropagationVolumeOnMaskFromGeodesicCenterOfFirstXYSlice
(iftImage *weight, iftFImage *cost, iftImage *pred, iftImage *root, iftImage *mask, iftLabeledSet *S)
{
  iftFHeap  *Q=NULL;
  int        i, p, q;
  iftVoxel   u, v;
  iftAdjRel *A=iftSpheric(sqrtf(3.0));
  float costval;

  iftMaximumValue(weight); 

  Q = iftCreateFHeap(weight->n, cost->val);

  for (p=0; p < weight->n; p++) {
    cost->val[p] = IFT_INFINITY_FLT;
    pred->val[p] = IFT_NIL;
    root->val[p] = p;
    if ((iftGetZCoord(weight,p)==0) && mask->val[p] > 0){
      cost->val[p]=0.0;    
    }
  }  

  // Selecting geodesic center of pores as seeds
  while(S != NULL) {
    p = S->elem;
    iftInsertFHeap(Q,p);
    S = S->next;
  }

  
  /* Geodesic path propagation by Image Foresting Transform */

  while (!iftEmptyFHeap(Q)) {
    p = iftRemoveFHeap(Q);
    u = iftGetVoxelCoord(weight, p);
    
    for (i = 1; i < A->n; i++)  {
      v = iftGetAdjacentVoxel(A, u, i);
      if (iftValidVoxel(weight, v)) {
	q   = iftGetVoxelIndex(weight, v);
	if(Q->color[q] != IFT_BLACK && mask->val[q] > 0) {
	  costval =  iftFCham3D_Hyd(cost,weight,p,q);
	  //costval =  MAX(cost->val[p], weight->val[q]); 
	  if (costval < cost->val[q]) {
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
  /*
  iftFMaximumValue(cost);
  printf("Fcost->max:%f\n",cost->maxval);
  */
  iftDestroyFHeap(&Q);
  iftDestroyAdjRel(&A);
  
  /*
  iftImage *tmp = iftCreateImage(cost->xsize,cost->ysize,cost->zsize);
  for (p = 0; p < cost->n; p++) {
		if ( cost->val[p] < 32766) {
			tmp->val[p] = (int)cost->val[p];
			}
		}
		
	iftWriteImage(tmp,"costint.scn");
	iftDestroyImage(&tmp);
	*/
}



void iftGeomPathPropagationVolumeOnMaskFromGeodesicCenterOfFirstXYSlice
(iftFImage *cost, iftImage *pred, iftImage *root, iftImage *mask, iftLabeledSet *S)
{
  iftFHeap  *Q=NULL;
  int        i, p, q;
  iftVoxel   u, v;
  iftAdjRel *A=iftSpheric(sqrtf(3.0));
  float costval;

  iftMaximumValue(mask); 

  Q = iftCreateFHeap(mask->n, cost->val);

  for (p=0; p < mask->n; p++) {
    cost->val[p] = IFT_INFINITY_FLT;
    pred->val[p] = IFT_NIL;
    root->val[p] = p;
    if ((iftGetZCoord(mask,p)==0) && mask->val[p] > 0){
      cost->val[p]=0.0;    
    }
  }  

  // Selecting geodesic center of pores as seeds
  while(S != NULL) {
    p = S->elem;
    iftInsertFHeap(Q,p);
    S = S->next;
  }

  
  /* Geodesic path propagation by Image Foresting Transform */

  while (!iftEmptyFHeap(Q)) {
    p = iftRemoveFHeap(Q);
    u = iftGetVoxelCoord(mask, p);
    
    for (i = 1; i < A->n; i++)  {
      v = iftGetAdjacentVoxel(A, u, i);
      if (iftValidVoxel(mask, v)) {
	q   = iftGetVoxelIndex(mask, v);
	if(Q->color[q] != IFT_BLACK && mask->val[q] > 0) {
	  costval =  iftFCham3D_Geo(cost,p,q);
	  if (costval < cost->val[q]) {
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

void iftDrawMinPathPerRoot(iftImage *orig, iftFImage *cost, iftImage *root, iftImage *pred, iftLabeledSet *roots)
{
  float *min_root_cost = NULL;
  int i, p, r, lb;
  iftVoxel u;
  int *root_id = NULL, nroots = 0, *end_nodes = NULL;
  iftLabeledSet *S = NULL;
  
  /*
  iftImage *tmp = iftCreateImage(cost->xsize,cost->ysize,cost->zsize);
  for (p = 0; p < cost->n; p++) {
		if ( cost->val[p] < 32766) {
			tmp->val[p] = (int)cost->val[p];
			}
		}
		
	iftWriteImage(tmp,"costint.scn");
	iftDestroyImage(&tmp);
	*/
  nroots = iftLabeledSetSize(roots);
  
  root_id = iftAllocIntArray(nroots);
  min_root_cost = iftAllocFloatArray(nroots);
  end_nodes = iftAllocIntArray(nroots);
  
  for(i = 0, S = roots; S != NULL; S = S->next, i++) {
    root_id[i] = S->elem;
    min_root_cost[i] = IFT_INFINITY_FLT;
    end_nodes[i] = IFT_NIL;
  }
  
  for (p = 0; p < cost->n; p++) {
    u = iftFGetVoxelCoord(cost, p);
    if (u.z == cost->zsize-1)	{
      r = root->val[p];
      
      for(i = 0; i < nroots; i++) {
	if(root_id[i] == r) {
	  if(cost->val[p] < min_root_cost[i]) {
	    min_root_cost[i] = cost->val[p];
	    end_nodes[i] = p;
	  }
	}
      }
    }
  }  

  lb = 0;
  for(i = 0; i < nroots; i++) {
    p = end_nodes[i];
    if(p != IFT_NIL) {
      lb = root_id[i]+1;
      while (p != IFT_NIL) {
	orig->val[p]=lb;
	p = pred->val[p];
      }
    }
  }
  fprintf(stderr,"lb:%d\n",lb);

  free(root_id);
  free(min_root_cost);
  free(end_nodes);
}


iftImage   *imfFImageToImage(iftFImage *img1, int Imax)
{
  iftImage *img2=iftCreateImage(img1->xsize,img1->ysize,img1->zsize);
  int p;
  float minval = IFT_INFINITY_FLT, maxval = IFT_INFINITY_FLT_NEG;

  float img1maxval = iftFMaximumValue(img1);
  float img1minval = iftFMinimumValue(img1);
  
  fprintf(stderr,"Imax:%d\n", Imax);
  fprintf(stderr,"min:%f max:%f\n", img1minval, img1maxval);
  
  if (img1maxval < img1minval){
    iftWarning("Image is empty","iftFImageToImage");
  }
  
    for (p=0; p < img2->n; p++) {
			if ( img1->val[p] < IFT_INFINITY_FLT ) {
				if ( img1->val[p] < minval ) minval = img1->val[p];
				if ( img1->val[p] > maxval ) maxval = img1->val[p];
			}
		}
  
  fprintf(stderr,"min:%f max:%f\n", minval, maxval);

    for (p=0; p < img2->n; p++) {
			if ( img1->val[p] < IFT_INFINITY_FLT )
				img2->val[p]=(int)(Imax*(img1->val[p]-minval)/(maxval-minval));
      }

  img2->dx = img1->dx;
  img2->dy = img1->dy;
  img2->dz = img1->dz;

	iftWriteImage(img2,"teste.scn");

  return(img2);
}



/* *************************************************************** */
/* *************************************************************** */
/* *************************************************************** */


int main(int argc, char *argv[]) 
{
  iftImage	 *bin_mask=NULL;
  iftFImage      *cost=NULL;
  iftImage *dist = NULL, *dist_comp = NULL; 
  iftImage       *root=NULL, *pred=NULL;
  iftImage       *orig = NULL;
  iftAdjRel      *A = NULL;
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
    iftError("Usage: iftTortuosity <orig_image.scn> <poresbin.mask>","main");
  
  t1 = iftTic();
  orig   = iftReadImage(argv[1]);
  bin_mask = iftReadImage(argv[2]);
  
  
  cost   = iftCreateFImage(bin_mask->xsize,bin_mask->ysize,bin_mask->zsize);
  pred   = iftCreateImage(bin_mask->xsize,bin_mask->ysize,bin_mask->zsize);
  root   = iftCreateImage(bin_mask->xsize,bin_mask->ysize,bin_mask->zsize);

  A = iftSpheric(sqrt(3.0));

  dist = iftEuclDistTrans(bin_mask, A, IFT_INTERIOR, NULL, NULL, NULL);
  dist_comp = iftComplement(dist);
  iftWriteImage(dist_comp, "dist_comp.scn");
  iftWriteImage(dist, "dist.scn");

  /*******************************************/
  iftImage *slice = NULL;
  iftImage *label = NULL;
  iftAdjRel *B = iftCircular(1.5);
  iftLabeledSet *S = NULL, *aux2 = NULL;

  slice = iftGetXYSlice(bin_mask, 0);
  label = iftRelabelRegions(slice, B);
  
  S = iftGeodesicCenters(label);

  // 2D seed coords to 3D (it shouldn't be necessary if slice is the first one, but just in case)
  for(aux2 = S; aux2 != NULL; aux2 = aux2->next) {
    iftVoxel v = iftGetVoxelCoord(label, aux2->elem);
    v.z = 0;
    aux2->elem = iftGetVoxelIndex(bin_mask, v);
  }
  iftDestroyAdjRel(&B);
  iftDestroyImage(&label);
  iftDestroyImage(&slice);

  /*******************************************/

	
  iftHydroPathPropagationVolumeOnMaskFromGeodesicCenterOfFirstXYSlice
  (dist_comp,cost,pred,root, bin_mask, S);
    
  //iftDrawMinPath(orig, cost, root, pred, iftMaximumValue(orig)+1);
  
  
		
  iftImage *aux = imfFImageToImage(cost,32766);
  //iftImage *aux = iftFImageToImage(cost,32766);
  iftWriteImage(aux,"cost_hyd.scn");
  iftDestroyImage(&aux);

  iftSetImage(orig, 0);
  iftDrawMinPathPerRoot(orig, cost, root, pred, S);
  iftWriteImage(orig,"path_bin_hyd.scn");

  iftWriteImage(root,"root_hyd.scn");
  iftWriteImage(pred,"pred_hyd.scn");
  
  
  /***************************************************************/
	/**									 Geometrical Computing 										**/
	/***************************************************************/
	
	iftDestroyImage(&dist);				 	//unused
  iftDestroyImage(&dist_comp);		//unused
	
	iftDestroyFImage(&cost);
  iftDestroyImage(&pred);
  iftDestroyImage(&root);
  cost   = iftCreateFImage(bin_mask->xsize,bin_mask->ysize,bin_mask->zsize);
  pred   = iftCreateImage(bin_mask->xsize,bin_mask->ysize,bin_mask->zsize);
  root   = iftCreateImage(bin_mask->xsize,bin_mask->ysize,bin_mask->zsize);

  iftGeomPathPropagationVolumeOnMaskFromGeodesicCenterOfFirstXYSlice
  (cost, pred, root, bin_mask, S);
  
  aux = imfFImageToImage(cost,32766);
  iftWriteImage(aux,"cost_geo.scn");
  iftDestroyImage(&aux);

  iftSetImage(orig, 0);
  iftDrawMinPathPerRoot(orig, cost, root, pred, S);
  iftWriteImage(orig,"path_bin_geo.scn");

  iftWriteImage(root,"root_geo.scn");
  iftWriteImage(pred,"pred_geo.scn");
  
  /***************************************************************/
  
  
  t2     = iftToc(); 
	
  fprintf(stdout,"Tortuosity computed in %f ms \n",iftCompTime(t1,t2));

  iftDestroyAdjRel(&A);
  iftDestroyImage(&bin_mask);
  iftDestroyFImage(&cost);
  iftDestroyImage(&pred);
  iftDestroyImage(&root);
  iftDestroyImage(&orig);
  iftDestroyImage(&dist);
  iftDestroyImage(&dist_comp);
  iftDestroyLabeledSet(&S);
  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   


  return(0);
}
