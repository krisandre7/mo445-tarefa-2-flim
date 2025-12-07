
#include "shared.h"


void UpdateLabelFromMask3(Scene *label, int l, Scene *mask){
  int p;
  for(p=0; p<label->n; p++){
    if(mask->data[p]==0 && label->data[p]==l)
      label->data[p] = 0;
    else if(mask->data[p]>0)
      label->data[p] = l;      
  }
}

void ClearLabelOutsideMask3(Scene *label, int l, Scene *mask){
  int p;
  for(p=0; p<label->n; p++)
    if(mask->data[p]==0 && label->data[p]==l)
      label->data[p] = 0;
}


void DrawMask3(Scene *scn, Scene *mask, int val){
  int p;
  for(p=0; p<scn->n; p++)
    if(mask->data[p]>0)
      scn->data[p] = val;
}


void PrintDots(int n){
  int i;

  for(i=0; i<n; i++)
    printf(".");
}


real CompSetMean(Set *S){
  Set *aux;
  int p,sum=0,n=0;
  real mean=0.0;

  aux = S;
  while(aux != NULL){
    p = aux->elem;
    sum += p;
    n++;
    aux = aux->next;
  }
  if(n>0)
    mean = ((real)sum/(real)n);
  return mean;
}


void WriteVoxelArray(Voxel *v, int n,
		     char *filename){
  FILE *fp;
  int i;

  fp = fopen(filename,"w");
  fprintf(fp,"%d\n",n);
  for(i=0; i<n; i++)
    fprintf(fp,"%d %d %d\n",v[i].x,v[i].y,v[i].z);
  fclose(fp);
}


void ReadVoxelArray(Voxel *v, int *n,
		    char *filename){
  FILE *fp;
  int i;

  fp = fopen(filename,"r");
  fscanf(fp,"%d",n);
  for(i=0; i<*n; i++)
    fscanf(fp,"%d %d %d",&v[i].x,&v[i].y,&v[i].z);
  fclose(fp);
}



void    DrawSet2Scene(Scene *scn, Set *S, 
		      int value){
  Set *aux;
  int p;

  aux = S;
  while(aux != NULL){
    p = aux->elem;
    scn->data[p] = value;
    aux = aux->next;
  }
}


Set    *Mask2Set3(Scene *mask){
  Set *S=NULL;
  int p,n;
  
  n = mask->xsize*mask->ysize*mask->zsize;
  for(p=0; p<n; p++){
    if(mask->data[p]>0)
      InsertSet(&S, p);
  }

  return S;
}



int   IntegerNormalize(int value,
		       int omin,int omax,
		       int nmin,int nmax){
  float tmp;
  int   i;
  if ( (omax - omin) == 0) return 0;
  tmp = ((float)(value - omin)) / ((float)(omax - omin));
  i = nmin + ROUND(tmp * ((float)(nmax - nmin)));
  return i;
}



int GetMarkerID(int mk){
  return (mk>>8); //(int)label/256;
}

unsigned char GetMarkerLabel(int mk){
  return (mk & 0xff);  //(label % 256);
}

int GetMarkerValue(int id, unsigned char lb){
  return ((id<<8) | lb);  //(id*256 + (int)lb);
}


BMap *SceneLabel2BMap(Scene *label){
  BMap *bmap;
  int n,p;

  n = label->xsize*label->ysize*label->zsize;
  bmap = BMapNew(n);
  for(p=0; p<n; p++){
    if(label->data[p]>0){
      _fast_BMapSet(bmap,p,1);
    }
    else{
      _fast_BMapSet(bmap,p,0);
    }
  }
  
  return bmap;
}

BMap *SceneMask2BMap(Scene *mask){
  return SceneLabel2BMap(mask);
}

BMap *SceneMarker2BMap(Scene *marker){
  unsigned char lb;
  BMap *bmap;
  int n,p;

  n = marker->xsize*marker->ysize*marker->zsize;
  bmap = BMapNew(n);
  for(p=0; p<n; p++){
    lb = GetMarkerLabel(marker->data[p]);
    if(lb>0){
      _fast_BMapSet(bmap,p,1);
    }
    else{
      _fast_BMapSet(bmap,p,0);
    }
  }
  
  return bmap;
}


void   CopyBMap2SceneMask(Scene *mask, BMap *bmap){
  int n,p;
  
  n = mask->n;
  if(n != bmap->N)
    Error("Incompatible sizes",
	  "CopyBMap2SceneMask");
  for(p=0; p<n; p++)
    mask->data[p] = _fast_BMapGet(bmap,p);
}


Scene *SceneMarker2Label(Scene *marker){
  unsigned char lb;
  Scene *label;
  int n,p;
  
  n = marker->xsize*marker->ysize*marker->zsize;
  label = CreateScene(marker->xsize,
		      marker->ysize,
		      marker->zsize);
  for(p=0; p<n; p++){
    lb = GetMarkerLabel(marker->data[p]);
    label->data[p] = lb;
  }
  return label;
}



Scene *MyLabelBinComp3(Scene *bin, AdjRel3 *A){
  Scene *label=NULL;
  int i,j,n,p,q,l=1;
  FIFOQ *Q=NULL;
  Voxel u,v;  

  label  = CreateScene(bin->xsize,bin->ysize,bin->zsize);
  n      = bin->n;

  Q = FIFOQNew(n);

  for (j=0; j < n; j++){
    if ((bin->data[j]==1)&&(label->data[j]==0)){
      label->data[j]=l;
      FIFOQPush(Q, j);

      while(!FIFOQEmpty(Q)){
	p = FIFOQPop(Q);

	u.x = VoxelX(bin, p);
	u.y = VoxelY(bin, p);
	u.z = VoxelZ(bin, p);
	for (i=1; i < A->n; i++){
	  v.x = u.x + A->dx[i];
	  v.y = u.y + A->dy[i];
	  v.z = u.z + A->dz[i];
	  if (ValidVoxel(bin,v.x,v.y,v.z)){
	    q = v.x + bin->tby[v.y] + bin->tbz[v.z];
	    if ((bin->data[q]==1)&&(label->data[q] == 0)){
	      label->data[q] = label->data[p];
	      FIFOQPush(Q, q);
	    }
	  }
	}
      }
      l++;
      FIFOQReset(Q);
    }
  }
  FIFOQDestroy(Q);

  return(label);
}


void SelectLargestComp(Scene *ero){
  AdjRel3 *A=NULL;
  Scene *label=NULL;
  int Lmax;
  int *area=NULL;
  int imax,i,p,n=ero->n;

  A = Spheric(1.8);
  label = MyLabelBinComp3(ero,A);
  Lmax = MaximumValue3(label);
  area = (int *)AllocIntArray(Lmax+1);

  for (p=0; p < n; p++)  
    if (label->data[p]>0)
      area[label->data[p]]++;
  imax = 0;
  for (i=1; i <= Lmax; i++) 
    if (area[i]>area[imax])
      imax = i;
  for (p=0; p < n; p++)  
    if (label->data[p]!=imax)
      ero->data[p]=0;

  DestroyScene(&label);
  DestroyAdjRel3(&A);
  free(area);
}


Set *SceneBorder(Scene *pred){
  Set *B=NULL;
  int x,y,z,p;

  z = 0;
  for (y=0; y < pred->ysize; y++) 
    for (x=0; x < pred->xsize; x++) {
      p = x + pred->tby[y] + pred->tbz[z];
      InsertSet(&B,p);
    }

  z = pred->zsize-1;
  for (y=0; y < pred->ysize; y++) 
    for (x=0; x < pred->xsize; x++) {
      p = x + pred->tby[y] + pred->tbz[z];
      InsertSet(&B,p);
    }

  y = 0;
  for (z=1; z < pred->zsize-1; z++) 
    for (x=0; x < pred->xsize; x++) {
      p = x + pred->tby[y] + pred->tbz[z];
      InsertSet(&B,p);
    }

  y = pred->ysize-1;
  for (z=1; z < pred->zsize-1; z++) 
    for (x=0; x < pred->xsize; x++) {
      p = x + pred->tby[y] + pred->tbz[z];
      InsertSet(&B,p);
    }

  x = 0;
  for (z=1; z < pred->zsize-1; z++) 
    for (y=1; y < pred->ysize-1; y++) {
      p = x + pred->tby[y] + pred->tbz[z];
      InsertSet(&B,p);
    }

  x = pred->xsize-1;
  for (z=1; z < pred->zsize-1; z++) 
    for (y=1; y < pred->ysize-1; y++) {
      p = x + pred->tby[y] + pred->tbz[z];
      InsertSet(&B,p);
    }

  return(B);
}



Scene *Mask2EDT3(Scene *bin, AdjRel3 *A, char side, int limit, char sign){
  Scene *Dx=NULL,*Dy=NULL,*Dz=NULL,*fbin,*fcont,*fcost,*cost;
  Queue *Q=NULL;
  int i,p,q,n,sz,xysize;
  Voxel u;
  int *sq=NULL,tmp=INT_MAX,dx,dy,dz;
  AdjVxl *N;
  AdjRel3 *A6 = Spheric(1.0);

  n  = MAX(bin->xsize,MAX(bin->ysize,bin->zsize));
  sq = AllocIntArray(n);
  for (i=0; i < n; i++) 
    sq[i]=i*i;

  sz = FrameSize3(A);  
  fbin = AddFrame3(bin,sz,0);
  fcont = GetBorder3(fbin,A6);
  fcost = AddFrame3(bin,sz,INT_MIN);
  Dx = CreateScene(fcost->xsize,fcost->ysize,fcost->zsize);
  Dy = CreateScene(fcost->xsize,fcost->ysize,fcost->zsize);  
  Dz = CreateScene(fcost->xsize,fcost->ysize,fcost->zsize);  
  N  = AdjVoxels(fcost,A);
  n  = fcost->xsize*fcost->ysize*fcost->zsize;
  Q = CreateQueue(2*sz*(bin->xsize+bin->ysize+bin->zsize)+3*sz*sz,n);

  switch (side) {
  case INTERIOR:
    for(p = 0; p < n; p++){
      if (fbin->data[p] != 0){
	if (fcont->data[p]>0){
	  fcost->data[p]=0;    
	  InsertQueue(Q,fcost->data[p]%Q->C.nbuckets,p);
	} else
	  fcost->data[p] = INT_MAX;	  
      }else{
	if (fcost->data[p]!=INT_MIN)
	  fcost->data[p] = 0;
      }
    }
    break;
  case EXTERIOR:
    for(p = 0; p < n; p++){
      if (fbin->data[p] == 0){
	if (fcost->data[p]!=INT_MIN)
	  fcost->data[p] = INT_MAX;	  
      }else{
	if (fcont->data[p]>0){
	  fcost->data[p]=0;    
	  InsertQueue(Q,fcost->data[p]%Q->C.nbuckets,p);
	}else
	  fcost->data[p] = 0;
      }
    }
    break;
  case BOTH:
  default:    
    for(p = 0; p < n; p++){
      if (fcont->data[p] > 0){
	fcost->data[p]=0;    
	InsertQueue(Q,fcost->data[p]%Q->C.nbuckets,p);
      }else{ 
	if (fcost->data[p]!=INT_MIN)
	  fcost->data[p]=INT_MAX;    
      }
    }
  }
  DestroyScene(&fcont);
  DestroyScene(&fbin);

  xysize = fcost->xsize * fcost->ysize;
  
  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    u.x = (p % xysize) % fcost->xsize;
    u.y = (p % xysize) / fcost->xsize;
    u.z = p / xysize;
    for (i=1; i < N->n; i++){
      q = p + N->dp[i];
      if (fcost->data[p] < fcost->data[q]){
	dx  = Dx->data[p] + abs(A->dx[i]);
	dy  = Dy->data[p] + abs(A->dy[i]);
	dz  = Dz->data[p] + abs(A->dz[i]);
	tmp = sq[dx] + sq[dy] + sq[dz];
	if (tmp < fcost->data[q] && tmp <= limit){
	  if (fcost->data[q] == INT_MAX)
	    InsertQueue(Q,tmp%Q->C.nbuckets,q);
	  else 
	    UpdateQueue(Q,q,fcost->data[q]%Q->C.nbuckets,tmp%Q->C.nbuckets);
	  fcost->data[q]  = tmp;	    
	  Dx->data[q] = dx;
	  Dy->data[q] = dy;
	  Dz->data[q] = dz;	  
	}
      }
    }
  }
  DestroyQueue(&Q);
  DestroyAdjVxl(&N);
  DestroyAdjRel3(&A6);
  cost = RemFrame3(fcost,sz);
  free(sq);
  DestroyScene(&Dx);
  DestroyScene(&Dy);
  DestroyScene(&Dz);
  DestroyScene(&fcost);

  // Eliminate infinite values */
  n = cost->n;
  for (i=0; i<n; i++) {
    if (cost->data[i]==INT_MAX)
      cost->data[i] = limit;
  }

  // sign scene
  if (sign != 0){
    n  = cost->n;
    if (side != INTERIOR)
      for (i=0; i<n; i++) {
	if (bin->data[i] == 0) {
	  cost->data[i] = -cost->data[i];
	}
      }
  }
  return(cost);
}


Scene *SceneMask2EDT3(Scene *bin, AdjRel3 *A, char side, int limit, char sign){
  float *Dx=NULL,*Dy=NULL,*Dz=NULL;
  Scene *fbin,*fcont,*fcost,*cost;
  Queue *Q=NULL;
  int i,p,q,n,sz,xysize;
  Voxel u;
  int tmp=INT_MAX,nbuckets;
  float dx,dy,dz,dmax;
  AdjVxl *N;
  AdjRel3 *A6 = Spheric(1.0);

  sz = FrameSize3(A);  
  fbin = AddFrame3(bin,sz,0);
  fcont = GetBorder3(fbin,A6);
  fcost = AddFrame3(bin,sz,INT_MIN);
  N  = AdjVoxels(fcost,A);
  n  = fcost->n;

  dmax = MAX(bin->dx, MAX(bin->dy, bin->dz));
  nbuckets  = ROUND(2*sz*dmax*(fcost->xsize*bin->dx+
			       fcost->ysize*bin->dy+
			       fcost->zsize*bin->dz)+
		    3*sz*sz*dmax*dmax)+1;
  Q = CreateQueue(nbuckets,n);
  Dx = AllocFloatArray(fcost->n);
  Dy = AllocFloatArray(fcost->n);
  Dz = AllocFloatArray(fcost->n);

  switch (side) {
  case INTERIOR:
    for(p = 0; p < n; p++){  
      if (fbin->data[p] != 0){
	if (fcont->data[p]>0){
	  fcost->data[p]=0;    
	  InsertQueue(Q,fcost->data[p]%Q->C.nbuckets,p);
	} else
	  fcost->data[p] = INT_MAX;	  
      }else{
	if (fcost->data[p]!=INT_MIN)
	  fcost->data[p] = 0;
      }
    }
    break;
  case EXTERIOR:
    for(p = 0; p < n; p++){
      if (fbin->data[p] == 0){
	if (fcost->data[p]!=INT_MIN)
	  fcost->data[p] = INT_MAX;	  
      }else{
	if (fcont->data[p]>0){
	  fcost->data[p]=0;    
	  InsertQueue(Q,fcost->data[p]%Q->C.nbuckets,p);
	}else
	  fcost->data[p] = 0;
      }
    }
    break;
  case BOTH:
  default:    
    for(p = 0; p < n; p++){
      if (fcont->data[p] > 0){
	fcost->data[p]=0;    
	InsertQueue(Q,fcost->data[p]%Q->C.nbuckets,p);
      }else{ 
	if (fcost->data[p]!=INT_MIN)
	  fcost->data[p]=INT_MAX;    
      }
    }
  }
  DestroyScene(&fcont);
  DestroyScene(&fbin);

  xysize = fcost->xsize * fcost->ysize;
  
  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    u.x = (p % xysize) % fcost->xsize;
    u.y = (p % xysize) / fcost->xsize;
    u.z = p / xysize;
    for (i=1; i < N->n; i++){
      q = p + N->dp[i];
      if (fcost->data[p] < fcost->data[q]){
	dx  = Dx[p] + abs(A->dx[i])*bin->dx;
	dy  = Dy[p] + abs(A->dy[i])*bin->dy;
	dz  = Dz[p] + abs(A->dz[i])*bin->dz;
	tmp = ROUND(dx*dx + dy*dy + dz*dz);
	if (tmp < fcost->data[q] && tmp <= limit){
	  if (fcost->data[q] == INT_MAX)
	    InsertQueue(Q,tmp%Q->C.nbuckets,q);
	  else 
	    UpdateQueue(Q,q,fcost->data[q]%Q->C.nbuckets,tmp%Q->C.nbuckets);
	  fcost->data[q]  = tmp;	    
	  Dx[q] = dx;
	  Dy[q] = dy;
	  Dz[q] = dz;	  
	}
      }
    }
  }
  DestroyQueue(&Q);
  DestroyAdjVxl(&N);
  DestroyAdjRel3(&A6);
  cost = RemFrame3(fcost,sz);
  free(Dx);
  free(Dy);
  free(Dz);
  DestroyScene(&fcost);

  // Eliminate infinite values */
  n = cost->n;
  for (i=0; i<n; i++) {
    if (cost->data[i]==INT_MAX)
      cost->data[i] = limit;
  }

  // sign scene
  if (sign != 0){
    n  = cost->n;
    if (side != INTERIOR)
      for (i=0; i<n; i++) {
	if (bin->data[i] == 0) {
	  cost->data[i] = -cost->data[i];
	}
      }
  }
  return(cost);
}


//--------------------------------------------

void BIA_RemoveDirectory(char *dest, char *src){
  int n,i;

  n = strlen(src);
  while(n>0){
    n--;
    if(src[n]=='/') break;
  }
  
  if(src[n]=='/') n++;

  i = 0;
  while(src[n]!='\0'){
    dest[i] = src[n];
    i++; n++;
  }
  dest[i] = '\0';
}


void StartTimer(timer *t){
  gettimeofday(t, NULL);
}


void StopTimer(timer *t){
  timer toc;
  float totaltime;

  gettimeofday(&toc,NULL);
  totaltime = (toc.tv_sec-t->tv_sec)*1000.0 + (toc.tv_usec-t->tv_usec)*0.001;

  if(totaltime<1000.0)
    printf("%.2f ms\n",totaltime);
  else if(totaltime<60.0*1000.0)
    printf("%.2f seg\n",totaltime/1000.0);
  else
    printf("%.2f min\n",totaltime/(60.0*1000.0));
}


//---------------------------------------
//---------------------------------------
//---------------------------------------

void MyRemoveDirectory(char *dest, char *src){
  int n,i;
#ifndef _WIN32
  char slash = '/';
#else
  char slash = '\\';
#endif

  n = strlen(src);
  while(n>0){
    n--;
    if(src[n]==slash) break;
    }
  
  if(src[n]==slash) n++;

  i = 0;
  while(src[n]!='\0'){
    dest[i] = src[n];
    i++; n++;
  }
  dest[i] = '\0';
}



Scene *EraseBackground(Scene *scn, Scene *mask, Set *obj, Set *bkg){
  Scene *grad,*cost,*label,*pred;
  AdjRel3 *A;
  GQueue *Q;
  Set *seed=NULL;
  Voxel u,v;
  int p,q,i,Imax,n,sz,tmp;

  A     = Spheric(1.0);
  grad  = MorphGrad3(scn,A);
  cost  = CreateScene(grad->xsize,grad->ysize,grad->zsize);
  pred  = CreateScene(grad->xsize,grad->ysize,grad->zsize);
  label = CopyScene(mask);
  SetScene(cost, INT_MAX);
  SetScene(pred, NIL);
  Imax = MaximumValue3(grad);
  sz   = grad->xsize*grad->ysize;
  n    = sz*grad->zsize;
  Q    = CreateGQueue(Imax+1,n,cost->data);

  //Restrict watershed inside mask:
  for(p=0; p<n; p++)
    if(mask->data[p]==0)
      cost->data[p] = 0;

  seed = obj;
  while (seed != NULL) {
    p = seed->elem;
    if (mask->data[p]==1){
      cost->data[p]  = 0;
      label->data[p] = 1;
      InsertGQueue(&Q, p);
    }
    seed = seed->next;
  }

  seed = bkg;
  while (seed != NULL) {
    p = seed->elem;
    if (mask->data[p]==1){
      cost->data[p]  = 0;
      label->data[p] = 0;
      InsertGQueue(&Q, p);
    }
    seed = seed->next;
  }

  while(!EmptyGQueue(Q)) {
    p = RemoveGQueue(Q);

    q = pred->data[p];
    if(q!=NIL && label->data[q]==0)
      label->data[p] = 0;

    i = p%sz;
    u.x = i%grad->xsize;
    u.y = i/grad->xsize;
    u.z = p/sz;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if (ValidVoxel(grad,v.x,v.y,v.z)){
	q = v.x + grad->tby[v.y] + grad->tbz[v.z];
	if (cost->data[p] < cost->data[q]){
	  tmp = MAX(cost->data[p],grad->data[q]);
	  if (tmp < cost->data[q]){
	    if (Q->L.elem[q].color == GRAY)
	      RemoveGQueueElem(Q,q);
	    cost->data[q] = tmp;
	    pred->data[q] = p;
	    InsertGQueue(&Q,q);
	  }
	}
      }
    }
  }
  DestroyGQueue(&Q);
  DestroyScene(&grad);
  DestroyScene(&cost);
  DestroyScene(&pred);
  DestroyAdjRel3(&A);

  return(label);
}


Scene *WatershedMask3(Scene *scn, Scene *mask, Set *obj, Set *bkg){
  Scene *grad,*cost,*label;
  AdjRel3 *A;
  GQueue *Q;
  Set *seed=NULL;
  Voxel u,v;
  int p,q,i,Imax,n,sz,tmp;

  A     = Spheric(1.0);
  grad  = MorphGrad3(scn,A);
  cost  = CreateScene(grad->xsize,grad->ysize,grad->zsize);
  label = CreateScene(grad->xsize,grad->ysize,grad->zsize);
  SetScene(label, 0);
  SetScene(cost, INT_MAX);
  Imax = MaximumValue3(grad);
  sz   = grad->xsize*grad->ysize;
  n    = sz*grad->zsize;
  Q    = CreateGQueue(Imax+1,n,cost->data);

  //Restrict watershed inside mask:
  for(p=0; p<n; p++)
    if(mask->data[p]==0)
      cost->data[p] = 0;

  seed = obj;
  while (seed != NULL) {
    p = seed->elem;
    if (mask->data[p]==1){
      cost->data[p]  = 0;
      label->data[p] = 1;
      InsertGQueue(&Q, p);
    }
    seed = seed->next;
  }

  seed = bkg;
  while (seed != NULL) {
    p = seed->elem;
    if (mask->data[p]==1){
      cost->data[p]  = 0;
      label->data[p] = 0;
      InsertGQueue(&Q, p);
    }
    seed = seed->next;
  }

  while(!EmptyGQueue(Q)) {
    p = RemoveGQueue(Q);

    i = p%sz;
    u.x = i%grad->xsize;
    u.y = i/grad->xsize;
    u.z = p/sz;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if (ValidVoxel(grad,v.x,v.y,v.z)){
	q = v.x + grad->tby[v.y] + grad->tbz[v.z];
	if (cost->data[p] < cost->data[q]){
	  tmp = MAX(cost->data[p],grad->data[q]);
	  if (tmp < cost->data[q]){
	    if (Q->L.elem[q].color == GRAY)
	      RemoveGQueueElem(Q,q);
	    cost->data[q] = tmp;
	    label->data[q] = label->data[p];
	    InsertGQueue(&Q,q);
	  }
	}
      }
    }
  }
  DestroyGQueue(&Q);
  DestroyScene(&grad);
  DestroyScene(&cost);
  DestroyAdjRel3(&A);

  return(label);
}


CImage *CopySubCImage(CImage *cimg, int j1, int i1, int j2, int i2){
  int i,j,p,q,ncols,nrows;
  CImage *sub=NULL;

  ncols = cimg->C[0]->ncols;
  nrows = cimg->C[0]->nrows;
  if( j1>=0 && i1 >=0 && j2<ncols && i2<nrows && j1<=j2 && i1<=i2 ){
    sub = CreateCImage(j2-j1+1, i2-i1+1);
    p=0;
    for(i=i1; i<=i2; i++){
      for(j=j1; j<=j2; j++){
	q = j + i*ncols;
	sub->C[0]->val[p] = cimg->C[0]->val[q];
	sub->C[1]->val[p] = cimg->C[1]->val[q];
	sub->C[2]->val[p] = cimg->C[2]->val[q];
	p++;
      }
    }
  }
  
  return sub;
}



void    ConvertCImageList2jpeg(FileList *L){
  char command[5000],file[2000];
  int i;

  for(i=0; i<L->n; i++){
    strcpy(file,GetFile(L, i));
    RemoveFileExtension(file);
    sprintf(command,"convert %s %s.jpg",GetFile(L, i),file);
    system(command);
  }
}


void    ConvertCImageList2ppm(FileList *L){
  char command[5000],file[2000];
  int i;

  for(i=0; i<L->n; i++){
    strcpy(file,GetFile(L, i));
    RemoveFileExtension(file);
    sprintf(command,"convert %s %s.ppm",GetFile(L, i),file);
    system(command);
  }
}


CImage *MergeCImages(CImage *A, CImage *B){
  int ncols,nrows;
  CImage *C;
  
  ncols = A->C[0]->ncols+B->C[0]->ncols;
  nrows = MAX(A->C[0]->nrows, B->C[0]->nrows);
  C = CreateCImage(ncols, nrows);
  PasteSubCImage(C, A,
		 0, nrows/2-A->C[0]->nrows/2);
  PasteSubCImage(C, B,
		 A->C[0]->ncols,
		 nrows/2-B->C[0]->nrows/2);
  return C;
}


void    MergeCImageList(FileList *in1,
			FileList *in2,
			FileList *out){
  CImage *A,*B,*C;
  int i;

  if(in1->n!=in2->n || in1->n!=out->n)
    Error("Incompatible FileLists",
	  "MergeCImageList");

  for(i=0; i<in1->n; i++){
    A = ReadCImage(GetFile(in1, i));
    B = ReadCImage(GetFile(in2, i));
    C = MergeCImages(A, B);
    WriteCImage(C, GetFile(out, i));
    DestroyCImage(&A);
    DestroyCImage(&B);
    DestroyCImage(&C);
  }
}



