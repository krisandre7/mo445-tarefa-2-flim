
#include "featmap.h"


FeatMap *CreateFeatMap(int n, int nfeat){
  FeatMap *fmap;
  int p;

  fmap = (FeatMap *)calloc(1, sizeof(FeatMap));
  if(fmap == NULL)
    Error(MSG1,"CreateFeatMap");
  fmap->n = n;
  fmap->nfeat = nfeat;

  fmap->data = (real **)calloc(n, sizeof(real *));
  if(fmap->data == NULL)
    Error(MSG1,"CreateFeatMap");
  for(p=0; p<n; p++)
    fmap->data[p] = AllocRealArray(nfeat);
  return fmap;
}


void    DestroyFeatMap(FeatMap **fmap){
  FeatMap *aux;
  int p;

  aux = *fmap;
  if(aux != NULL){
    if(aux->data != NULL){
      for(p=0; p<aux->n; p++)
	free((aux->data)[p]);
      free(aux->data);
    }
    free(aux);
    *fmap = NULL;
  }
}


/*Creates a copy of the original "FeatMap".*/
FeatMap *CloneFeatMap(FeatMap *fmap){
  FeatMap *clone;
  int i;
  clone = CreateFeatMap(fmap->n, fmap->nfeat);
  for(i=0; i<fmap->n; i++)
    memcpy(clone->data[i],
	   fmap->data[i],
	   fmap->nfeat*sizeof(real));

  return clone;
}


FeatMap *Scene2FeatMap(Scene *scn){
  FeatMap *fmap;
  float mean,stdev;
  real *fv;
  int n,p,val;

  n = scn->xsize*scn->ysize*scn->zsize;
  ComputeDescriptiveStatistics(scn, &mean, &stdev);
  fmap = CreateFeatMap(n, 1);

  for(p=0; p<n; p++){
    val = scn->data[p];
    fv  = fmap->data[p];
    fv[0] = ((real)val - mean)/stdev;
  }

  return fmap;
}


FeatMap *Scene2FeatMapByLocalIFT(Scene *scn, Scene *mask, int K){
  FeatMap *fmap;
  Scene *cost;
  AdjRel3 *A=Spheric(1.0);
  GQueue *Q=NULL;
  Voxel u,v;
  real *fv;
  int Imax,edge,cst;
  int n,p,s,t,i,ki,val_p;
  int *px = (int *)calloc(K,sizeof(int));
  float mean,stdev;

  n    = scn->xsize*scn->ysize*scn->zsize;
  ComputeDescriptiveStatistics(scn, &mean, &stdev);
  fmap = CreateFeatMap(n, 1);
  cost = CreateScene(scn->xsize, scn->ysize, scn->zsize);
  SetScene(cost, INT_MAX);
  Imax = MaximumValue3(scn);
  Q    = CreateGQueue(Imax+1, n, cost->data);

  for(p=0; p<n; p++){
    if(mask->data[p]==0) continue;
    val_p = scn->data[p];
    fv    = fmap->data[p];
    cost->data[p] = 0;
    InsertGQueue(&Q,p);

    for(ki=0; ki<K; ki++){
      s = RemoveGQueue(Q);
      fv[0] += ((float)scn->data[s]-mean)/stdev;
      px[ki] = s;
      u.x = VoxelX(scn, s);
      u.y = VoxelY(scn, s);
      u.z = VoxelZ(scn, s);
      for(i=1; i<A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	v.z = u.z + A->dz[i];
	if(ValidVoxel(scn, v.x, v.y, v.z)){
	  t = VoxelAddress(scn, v.x, v.y, v.z);
	  if(Q->L.elem[t].color != BLACK){	  
	    edge = abs(scn->data[t] - val_p);
	    cst  = cost->data[s]+edge; //MAX(cost->data[s],edge);
	    if(cst < cost->data[t]){
	      if(Q->L.elem[t].color == GRAY)
		RemoveGQueueElem(Q,t); // color of t = BLACK
	      cost->data[t] = cst;
	      InsertGQueue(&Q,t);
	    }
	  }
	}
      }
    }
    fv[0] /= (float)K;

    /* Reset queue manually */
    for(ki=0; ki<K; ki++){
      s = px[ki];
      cost->data[s] = INT_MAX;
      Q->L.elem[s].color = WHITE;
    }
    while (!EmptyGQueue(Q)){
      s = RemoveGQueue(Q);
      cost->data[s] = INT_MAX;
      Q->L.elem[s].color = WHITE;
    }
    Q->C.minvalue = INT_MAX;
    Q->C.maxvalue = INT_MIN;
  }

  DestroyScene(&cost);
  DestroyAdjRel3(&A);
  DestroyGQueue(&Q);
  free(px);

  return fmap;
}


float GetMedianValue(int *vec, int i, int j){
  float median;
  int s;
  
  s = j-i+1;
  if(s%2==0)
    median = (vec[i+s/2-1] + vec[i+s/2])/2.0;
  else
    median = vec[i+s/2];
  
  return median;
}


FeatMap *Scene2FeatMapBy5NumSummary(Scene *scn, Scene *mask, float radius){
  FeatMap *fmap;
  AdjRel3 *A=Spheric(radius);
  Voxel u,v;
  real *fv;
  int *vec;
  int n,p,q,i,s;
  float mean,stdev;

  n = scn->xsize*scn->ysize*scn->zsize;
  ComputeDescriptiveStatistics(scn, &mean, &stdev);
  fmap = CreateFeatMap(n, 5);
  vec = AllocIntArray(A->n);

  for(p=0; p<n; p++){
    if(mask->data[p]==0) continue;
    u.x = VoxelX(scn, p);
    u.y = VoxelY(scn, p);
    u.z = VoxelZ(scn, p);    
    for(i=0; i<A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if(ValidVoxel(scn, v.x, v.y, v.z)){
	q = VoxelAddress(scn, v.x, v.y, v.z);
	vec[i] = scn->data[q];
      }
      else
	vec[i] = scn->data[p];
    }
    SelectionSort(vec, A->n, INCREASING);
    fv = fmap->data[p];
    fv[0] = (real)vec[0];      //Minimum.
    fv[4] = (real)vec[A->n-1]; //Maximum.

    s = A->n;
    if(s%2==0){
      fv[2] = (real)(vec[s/2-1] + vec[s/2])/2.0; //Median.
      fv[1] = GetMedianValue(vec, 0, s/2-1);
      fv[3] = GetMedianValue(vec, s/2, A->n-1);
    }
    else{
      fv[2] = (real)vec[s/2];                    //Median.
      fv[1] = GetMedianValue(vec, 0, s/2);
      fv[3] = GetMedianValue(vec, s/2, A->n-1);      
    }

    for(i=0; i<5; i++)
      fv[i] = (fv[i]-mean)/stdev;
  }
  DestroyAdjRel3(&A);
  free(vec);

  return fmap;
}


FeatMap *Scene2FeatMapByKNN(Scene *scn, Scene *mask, int K, float radius){
  FeatMap *fmap;
  AdjRel3 *A=Spheric(radius);
  Voxel u,v;
  real *fv;
  int *vec;
  int n,p,q,i,j,val_p,d1,d2,l;
  float mean,stdev;

  n = scn->xsize*scn->ysize*scn->zsize;
  ComputeDescriptiveStatistics(scn, &mean, &stdev);
  fmap = CreateFeatMap(n, K);
  vec = AllocIntArray(A->n);

  for(p=0; p<n; p++){
    if(mask->data[p]==0) continue;
    val_p = scn->data[p];
    u.x = VoxelX(scn, p);
    u.y = VoxelY(scn, p);
    u.z = VoxelZ(scn, p);    
    for(i=0; i<A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if(ValidVoxel(scn, v.x, v.y, v.z)){
	q = VoxelAddress(scn, v.x, v.y, v.z);
	vec[i] = scn->data[q];
      }
      else
	vec[i] = scn->data[p];
    }
    SelectionSort(vec, A->n, INCREASING);

    for(i=0; i<A->n; i++)
      if(vec[i]==val_p) break;

    j = i;
    for(l=1; l<K; l++){
      if(i>0) d1 = abs(val_p-vec[i-1]);
      else    d1 = INT_MAX;

      if(j<A->n-1) d2 = abs(val_p-vec[j+1]);
      else         d2 = INT_MAX;
      
      if(d1<d2) i--;
      else      j++;
    }

    fv = fmap->data[p];
    for(l=0; l<K; l++)
      fv[l] = ((float)vec[i+l]-mean)/stdev;
  }

  DestroyAdjRel3(&A);
  free(vec);

  return fmap;
}



FeatMap *Scene2FeatMapByMaxMinVal(Scene *scn, Scene *mask, float radius){
  FeatMap *fmap;
  AdjRel3 *A=Spheric(radius);
  Voxel u,v;
  real *fv;
  int n,p,q,i,val,val_p,min,max;
  float mean,stdev;

  n = scn->xsize*scn->ysize*scn->zsize;
  ComputeDescriptiveStatistics(scn, &mean, &stdev);
  fmap = CreateFeatMap(n, 3);

  for(p=0; p<n; p++){
    if(mask->data[p]==0) continue;
    u.x = VoxelX(scn, p);
    u.y = VoxelY(scn, p);
    u.z = VoxelZ(scn, p);    

    val_p = scn->data[p];
    min = val_p;
    max = val_p;
    for(i=1; i<A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if(ValidVoxel(scn, v.x, v.y, v.z)){
	q = VoxelAddress(scn, v.x, v.y, v.z);
	if(mask->data[q]==0) continue;
	val = scn->data[q];
	if(val<min) min = val;
	if(val>max) max = val;
      }
    }
    fv  = fmap->data[p];
    fv[0] = ((real)val_p - mean)/stdev;
    fv[1] = ((real)min   - mean)/stdev;
    fv[2] = ((real)max   - mean)/stdev;
  }
  DestroyAdjRel3(&A);

  return fmap;
}



real DistanceSub(real *fv1, real *fv2, int nfeat){
  real d=0.0;
  int i;

  for(i=0; i<nfeat; i++)
    d += fv2[i] - fv1[i];

  return d;
}


real DistanceL1(real *fv1, real *fv2, int nfeat){
  real d=0.0, aux;
  int i;

  for(i=0; i<nfeat; i++){
    aux = fv1[i] - fv2[i];
    if(aux<0)
      d += -aux;
    else
      d += aux;
  }

  return d;
}


real DistanceL2(real *fv1, real *fv2, int nfeat){
  real aux, d=0.0;
  int i;

  for(i=0; i<nfeat; i++){
    aux = fv1[i] - fv2[i];
    d += aux*aux;
  }

  return (real)sqrtf(d);
}


//DistanceSum
real DistanceGRAD(real *fv1, real *fv2, int nfeat){
  real d=0.0;
  int i;

  for(i=0; i<nfeat; i++)
    d += fv1[i]+fv2[i];
  d /= (2.0*nfeat);

  return (real)d;
}



//Converte as caracteristicas do volume em um complete 
//graph ordenado pelo label.
ComplGraph *FeatMap2ComplGraph(FeatMap *fmap, Scene *label){
  ComplGraph *cg;
  real *fv;
  int lb,j,i,c,p,n,nclasses;

  n = nclasses = 0;
  for(p=0; p<fmap->n; p++){
    lb = label->data[p];
    if(lb>0)        n++;
    if(lb>nclasses) nclasses = lb;
  }

  cg = CreateComplGraph(n, nclasses, fmap->nfeat);

  i = 0;
  for(c=1; c<=cg->nclasses; c++){
    cg->nclass[c] = 0;
    for(p=0; p<fmap->n; p++){
      if(label->data[p]==c){
	fv = fmap->data[p];
	for(j=0; j<fmap->nfeat; j++)
	  cg->node[i].data[j] = fv[j];
	cg->node[i].label = c;
	cg->nclass[c]++;
	i++;
      }
    }
  }

  return cg;
}

