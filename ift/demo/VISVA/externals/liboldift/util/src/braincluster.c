#include "ift.h"

typedef struct _arcweights {
  float **val;
  int     n;
} ArcWeights;

ArcWeights *CreateArcWeights(int nlabels)
{
  ArcWeights *oindex=(ArcWeights *)calloc(1,sizeof(ArcWeights));
  int i;

  oindex->val = (float **) calloc(nlabels,sizeof(float *));
  for (i=0; i < nlabels; i++) 
    oindex->val[i] = (float *)calloc(nlabels,sizeof(float));
  
  oindex->n = nlabels;
  return(oindex);
}

void DestroyArcWeights(ArcWeights **oindex)
{
  int i;

  if ((*oindex) != NULL){
    for (i=0; i < (*oindex)->n; i++) 
      free((*oindex)->val[i]);
    free((*oindex)->val);
    free((*oindex));
    *oindex = NULL;
  }
}

void PrintArcWeights(ArcWeights *oindex)
{
  int i,j;

  printf("\n");

  for(i=0; i < oindex->n; i++) {
    for(j=0; j < oindex->n; j++) 
      printf("%5.2f ",oindex->val[i][j]);
    printf("\n");
  }
}

ArcWeights *OverlappingIndex(Subgraph *sg){ 
  int   i, j, k;
  float weight,tot,max;
  ArcWeights *oindex;

  oindex = CreateArcWeights(sg->nlabels);

  for (i = 0; i < sg->nnodes; i++){
    for (j = 0; (j < sg->nnodes); j++){
      k = sg->ordered_list_of_nodes[j];
      weight = opf_ArcWeight(sg->node[k].feat,sg->node[i].feat,sg->nfeats);

      if (weight <= sg->node[k].radius){
	oindex->val[sg->node[i].label][sg->node[k].label]++;
      }
    }
  }
  
  // Normalize the overlapping index

  for(i=0; i < sg->nlabels; i++){
    tot=0;
    for(j=0; j < sg->nlabels; j++) 
      tot += oindex->val[i][j];
    for(j=0; j < sg->nlabels; j++) 
      oindex->val[i][j] /= tot;
  }

  return(oindex);
}


int FindSubgraphRoot(Subgraph *sg, int i)
{
  if (sg->node[i].root == i)
    return(i);
  else
    return(sg->node[i].root=FindSubgraphRoot(sg, sg->node[i].root));
}

void MergeOverlapClusters(Subgraph *sg, ArcWeights *oindex)
{
  int i,j;

  printf("initial number of clusters %d\n",sg->nlabels);

  for (i=0; i < sg->nnodes; i++) {
    for (j=0; j < sg->nnodes; j++) {
      if (sg->node[i].label != sg->node[j].label)
	if (oindex->val[sg->node[i].label][sg->node[j].label] >= 0.10){
	  sg->node[i].root = FindSubgraphRoot(sg,j);
	}
    }
  }

  for (i=0; i < sg->nnodes; i++) 
    sg->node[i].root = FindSubgraphRoot(sg,i);

  j=0;
  for (i=0; i < sg->nnodes; i++) 
    if (i==sg->node[i].root){
      sg->node[sg->node[i].root].label=j;
      j++;
    }
  sg->nlabels=j;
  
  for (i=0; i < sg->nnodes; i++) 
    sg->node[i].label = sg->node[sg->node[i].root].label;

  printf("final number of clusters %d\n",sg->nlabels);
  
}


Scene *TreatOutliers3(Scene *label, Scene *mask)
{
  Scene *flabel=CopyScene(label);
  int p,q,r,s,i,n=label->xsize*label->ysize*label->zsize;
  int aux,xysize=label->xsize*label->ysize;
  Voxel u,v;
  AdjRel3 *A=Spheric(1.0);
  int *FIFO,first=0,last=0;
  char *color;
  Set *S=NULL;

  FIFO   = AllocIntArray(n);
  color  = AllocCharArray(n);

  for(p=0; p < n; p++) {
    if (mask->data[p] != 0 && flabel->data[p]==0){
	FIFO[last]=p;
	last++;
	InsertSet(&S,p);
	color[p]=GRAY;
	while(first != last){
	  q = FIFO[first];
	  color[q] = BLACK;
	  aux = q%xysize;
	  u.x = aux%label->xsize;
	  u.y = aux/label->xsize;
	  u.z = q/xysize;
	  first++;
	  for (i=1; i < A->n; i++) {
	    v.x = u.x + A->dx[i];
	    v.y = u.y + A->dy[i];
	    v.z = u.z + A->dz[i];
	    if (ValidVoxel(label,v.x,v.y,v.z)){
	      r = v.x + label->tby[v.y] + label->tbz[v.z];
	      if (mask->data[r]!=0){
		if (color[r]==WHITE){
		  if (label->data[r]==0){
		    FIFO[last] = r;
		    last++;
		    color[r]=GRAY;
		    InsertSet(&S,r);
		  }else{
		    flabel->data[p] = label->data[r];
		    first=last=0;
		    break;
		  }
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
  DestroyAdjRel3(&A);
  free(FIFO);
  free(color);

  return(flabel);
}


Scene *SceneClassifyKnnGraph( Subgraph *sg, Scene *mask, Features3 *f ) {
  float dist;
  int   i,p,k;
  Scene *label = CreateScene( f->xsize, f->ysize, f->zsize );

  for( p = 0; p < f->nelems; p++ ) {
    if( ( mask == NULL ) || ( mask->data[ p ] != 0 ) ) {
      for( i = 0; i < sg->nnodes; i++ ) {
      	k = sg->ordered_list_of_nodes[ i ];
	dist = opf_ArcWeight( sg->node[ k ].feat, f->elem[ p ].feat, sg->nfeats );
	if( dist <= sg->node[ k ].radius ) {
	  label->data[ p ] = sg->node[ k ].label+1;
	  break;
	}
      }
    }
  }

  return( label );
}

void SeparateWMClusters(Scene *scn, Scene *flabel, Scene **wm, Scene **mask)
{
  int nlabels=MaximumValue3(flabel);
  double *mean=AllocDoubleArray(nlabels+1);
  double *nelems=AllocDoubleArray(nlabels+1),Vmax;
  int i,p,n=scn->xsize*scn->ysize*scn->zsize,imax;
  
  (*wm)=CreateScene(scn->xsize,scn->ysize,scn->zsize);
  (*mask)=CreateScene(scn->xsize,scn->ysize,scn->zsize);
  
  for (p=0; p < n; p++) {
    if (flabel->data[p]>0){
      mean[flabel->data[p]]+= scn->data[p];
      nelems[flabel->data[p]]++;
    }
  }

  for (i=1; i <= nlabels; i++) {
    mean[i] /= nelems[i];
  }

  Vmax=-FLT_MAX;
  imax=NIL;
  for (i=1; i <= nlabels; i++) {
    printf("mean %lf nelems %lf\n",mean[i],nelems[i]);
    if ((mean[i]>Vmax)&&(nelems[i]>10000)){
      Vmax = mean[i];
      imax = i;
    }
  }

  for (p=0; p < n; p++) {
    if (flabel->data[p]==imax)
      (*wm)->data[p]=1;
    else{
      if (flabel->data[p]>0){
	if (mean[flabel->data[p]]>=Vmax)
	  (*wm)->data[p]=1;
	else
	  (*mask)->data[p]=1;
      }
    }
  }
  free(mean); free(nelems);
}

Scene *PutAllTogether(Scene *scn, Scene *flabel, Scene *wm)
{
  Scene *label=CreateScene(scn->xsize,scn->ysize,scn->zsize);
  int nlabels=MaximumValue3(flabel);
  double *mean=AllocDoubleArray(nlabels+1);
  double *nelems=AllocDoubleArray(nlabels+1),Vmin;
  int i,p,n=scn->xsize*scn->ysize*scn->zsize,imin;
  
  
  for (p=0; p < n; p++) {
    if (flabel->data[p]>0){
      mean[flabel->data[p]]+= scn->data[p];
      nelems[flabel->data[p]]++;
    }
  }

  for (i=1; i <= nlabels; i++) {
    mean[i] /= nelems[i];
  }

  Vmin=FLT_MAX;
  imin=NIL;
  for (i=1; i <= nlabels; i++) {
    if ((mean[i]<Vmin)&&(nelems[i]>10000)){
      Vmin = mean[i];
      imin = i;
    }
  }

  for (p=0; p < n; p++) {
    if (wm->data[p]==1)
      label->data[p]=3;
    if (flabel->data[p]==imin)
      label->data[p]=1;
    else{
      if (flabel->data[p]>0){
	if (mean[flabel->data[p]]<=Vmin)
	  label->data[p]=1;
	else
	  label->data[p]=2;
      }
    }
  }
  free(mean); free(nelems);
  return(label);
}


int main(int argc, char **argv)
{
  timer     *t1=NULL,*t2=NULL;
  Scene     *scn=NULL,*mask=NULL,*label=NULL,*flabel=NULL;
  Scene     *wm=NULL;
  Subgraph  *sg=NULL;
  Features3 *f=NULL;
  ArcWeights *oindex=NULL;

  opf_ArcWeight=opf_EuclDist;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);
  struct mallinfo info;
  int MemDinInicial, MemDinFinal;
  free(trash);
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=5)
    Error("Usage must be: braincluster <scene> <mask> <kmax1> <kmax2>","main");


  t1 = Tic();

  scn  = ReadScene(argv[1]);
  mask = ReadScene(argv[2]);
  sg   = RandomSampl3(scn,mask,1000);
  f     = MedianSceneFeats(scn,mask,1.8);
  SetSubgraphFeatures3(sg,f);
  opf_BestkMinCut(sg,1,atoi(argv[3]));
  opf_OPFClustering(sg);
  
  oindex = OverlappingIndex(sg);
  PrintArcWeights(oindex);
  MergeOverlapClusters(sg, oindex);
  label=SceneClassifyKnnGraph(sg,mask,f);
  flabel = TreatOutliers3(label,mask);
  WriteScene(flabel,"firstlabels.scn.bz2");
  DestroyScene(&mask);
  SeparateWMClusters(scn,flabel,&wm,&mask);
  WriteScene(wm,"wm.scn.bz2");

  //exit(0);

  DestroySubgraph(&sg);
  DestroyScene(&flabel);
  DestroyScene(&label);
  DestroyFeatures3(&f);
  DestroyArcWeights(&oindex);
  sg   = RandomSampl3(scn,mask,1000);
  f    = CoOccur3(scn,mask,3);
  SetSubgraphFeatures3(sg,f);
  opf_BestkMinCut(sg,1,atoi(argv[4]));
  opf_OPFClustering(sg);
  
  oindex = OverlappingIndex(sg);
  PrintArcWeights(oindex);
  MergeOverlapClusters(sg, oindex);
  label=SceneClassifyKnnGraph(sg,mask,f);
  flabel = TreatOutliers3(label,mask);
  DestroyScene(&label);
  label = PutAllTogether(scn,flabel,wm);

  WriteScene(label,"finallabels.scn.bz2");
  
  DestroyScene(&scn);
  DestroyScene(&wm);
  DestroySubgraph(&sg);
  DestroyScene(&mask);
  DestroyScene(&flabel);
  DestroyScene(&label);
  DestroyFeatures3(&f);
  DestroyArcWeights(&oindex);

  t2 = Toc();

  fprintf(stdout,"scenecluster in %f ms\n",CTime(t1,t2));

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);

  return(0);
}
