



#include <oldift.h>
#include "measures.h"
#include "rigidregistration3.h"



Measures* CreateMeasures()
{
  Measures *M = (Measures *) malloc(sizeof(Measures));
  if(M == NULL)
    Error(MSG1,"CreateMeasures");
  M->n = 0;
  M->measure=NULL;
  return M;
}


void DestroyMeasures(Measures **M){
  Measures *aux;
  aux = *M;
  if (aux != NULL){
    free(aux->measure);
    free(aux);
    *M = NULL;
  }
}


void AddMeasure(Measures *M, char *name, float value) 
{
  measure_node *list = (measure_node *) calloc(M->n+1,sizeof(measure_node));
  if (M->n>0) memcpy(list,M->measure, M->n*sizeof(measure_node));
  strcpy(list[M->n].name, name);
  list[M->n].value=value;
  free(M->measure);
  M->measure=list;
  M->n++;
}

int SearchMeasureByName(Measures *M, char *name, float *value) 
{
  int i;
  for (i=0; i<M->n; i++) 
    if (strcmp(name,M->measure[i].name)==0) {
      *value = M->measure[i].value;
      return 1;
    }
  return 0;  // not found
}


void PrintMeasuresList(Measures *M) 
{
  int i;
  printf("MEASURES LIST:\n");
  for (i=0; i<M->n; i++)
    printf("Name: %s  -  Value=%f\n",M->measure[i].name,M->measure[i].value);
}





double KLMeasure(Curve *histref, Curve *histimg)
{
	int i,nbins=histref->n;
	double kl=0;

	for (i=0;i<=nbins-1;i++)
		if (histimg->Y[i]!=0 && histref->Y[i]!=0)
			kl = kl + histimg->Y[i]*log(histimg->Y[i]/histref->Y[i]);
	return kl;
}

void ComputeM1(Measures *M, Scene *scn)
{
  Curve *histL,*histR;
  Scene *left=GetLeftAxialHalf(scn);
  Scene *tmp=GetRightAxialHalf(scn);
  Scene *right=FlipSceneAxial(tmp);
  DestroyScene(&tmp);
  histL=NormHistogram3(left);
  histR=NormHistogram3(right);
  float m1= KLMeasure(histL,histR);
  DestroyCurve(&histL);
  DestroyCurve(&histR);
  AddMeasure(M,"M1_GLOBAL",m1);

  int dx[3],dy[5],dz[5];
  dx[0]=0; dx[1]=left->xsize/2; dx[2]=left->xsize-1;
  dy[0]=0; dy[1]=left->ysize/4; dy[2]=left->ysize/2; dy[3]=(3*left->ysize)/4 ; dy[4]=left->ysize;
  dz[0]=0; dz[1]=left->zsize/4; dz[2]=left->zsize/2; dz[3]=(3*left->zsize)/4 ; dz[4]=left->zsize;

  int x,y,z;
  Scene *left_p,*right_p;
  char str[30];
  int counter=1;
  for (x=0;x<=1;x++)
    for (y=0;y<=3;y++)
      for (z=0;z<=3;z++) {
	left_p = ROI3(left,dx[x],dy[y],dz[z],dx[x+1]-1,dy[y+1]-1,dz[z+1]-1);
	right_p = ROI3(right,dx[x],dy[y],dz[z],dx[x+1]-1,dy[y+1]-1,dz[z+1]-1);
	histL=NormHistogram3(left_p);
	histR=NormHistogram3(right_p);
	m1 = KLMeasure(histL,histR);
	DestroyCurve(&histL);
	DestroyCurve(&histR);
	sprintf(str,"M1_%03d",counter++);
	AddMeasure(M,str,m1);
	DestroyScene(&left_p);
	DestroyScene(&right_p);
      }
  DestroyScene(&left);
  DestroyScene(&right);

}





float M2_TextGrad_Distance(float *f1, float *f2, int n){
  int i;
  float dist;
  
  dist = 0;
  for (i=0; i < n; i++) 
    dist += (f2[i]-f1[i]);///2.0;

  //  dist /= n;
  return(dist);//exp(-(dist-0.5)*(dist-0.5)/2.0));
}


Scene* M2_TextGradient3(Scene *scn){
  float   dist,gx,gy,gz;  
  int     Imax,i,p,q,n=scn->xsize*scn->ysize*scn->zsize;
  Voxel   u,v;
  AdjRel3 *A=Spheric(1.0),*A6=Spheric(1.0);
  float   *mg=AllocFloatArray(A6->n);
  Scene   *grad=CreateScene(scn->xsize,scn->ysize,scn->zsize);

  typedef struct _features {
    float *f;
  } Features;

  Features *feat=(Features *)calloc(n,sizeof(Features));
  for (p=0; p < n; p++)
    feat[p].f = AllocFloatArray(A->n);

  Imax = MaximumValue3(scn);

  for (u.z=0; u.z < scn->zsize; u.z++) 
    for (u.y=0; u.y < scn->ysize; u.y++) 
      for (u.x=0; u.x < scn->xsize; u.x++) {
	p = u.x + scn->tby[u.y] + scn->tbz[u.z];
	for (i=0; i < A->n; i++) {
	  v.x = u.x + A->dx[i];
	  v.y = u.y + A->dy[i];
	  v.z = u.z + A->dz[i];
	  if (ValidVoxel(scn,v.x,v.y,v.z)){
	    q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	    feat[p].f[i]=(float)scn->data[q];///(float)Imax;
	  }
	}
      }

  for (i=0; i < A6->n; i++) 
    mg[i]=sqrt(A6->dx[i]*A6->dx[i]+A6->dy[i]*A6->dy[i]+A6->dz[i]*A6->dz[i]);

  for (u.z=0; u.z < scn->zsize; u.z++) 
    for (u.y=0; u.y < scn->ysize; u.y++) 
      for (u.x=0; u.x < scn->xsize; u.x++) {
	p = u.x + scn->tby[u.y] + scn->tbz[u.z];
	gx = gy = gz = 0.0;
	for (i=1; i < A6->n; i++) {
	  v.x = u.x + A6->dx[i];
	  v.y = u.y + A6->dy[i];
	  v.z = u.z + A6->dz[i];
	  if (ValidVoxel(scn,v.x,v.y,v.z)){
	    q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	    dist = M2_TextGrad_Distance(feat[p].f,feat[q].f,A->n);
	    gx  += dist*A6->dx[i]/mg[i]; 
	    gy  += dist*A6->dy[i]/mg[i]; 
	    gz  += dist*A6->dz[i]/mg[i]; 
	  }
	} 
	grad->data[p]=(int)sqrt(gx*gx + gy*gy + gz*gz);
      }


  for (p=0; p < n; p++)
    free(feat[p].f);
  free(feat);

  free(mg);
  DestroyAdjRel3(&A);
  DestroyAdjRel3(&A6);
  return(grad);
}


void M2_ZeroBorders(Scene *scn, int bordersize)
{
  Voxel u;
  int p;
  if (bordersize<1) return;
  for (u.z=0; u.z < scn->zsize; u.z++) 
    for (u.y=0; u.y < scn->ysize; u.y++)
      for (u.x=0; u.x < scn->xsize; u.x++) {
	if (u.x<bordersize || u.y<bordersize || u.z<bordersize ||
	    u.x>=scn->xsize-bordersize || u.y>=scn->ysize-bordersize || u.z>=scn->zsize-bordersize) {
	  p = u.x + scn->tby[u.y] + scn->tbz[u.z];
	  scn->data[p]=0;
	}
      }
}


Scene* M2_BinarizeByThreshold(Scene *scn, float threshold)
{						
  Scene *bin;
  int i;
  int n = scn->xsize * scn->ysize * scn->zsize; 
  bin = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  for (i=0;i<n;i++)
    if (scn->data[i]>threshold) bin->data[i]=1;
  bin->dx=scn->dx;
  bin->dy=scn->dy;
  bin->dz=scn->dz;
  return bin;
}



int M2_ThresholdToGetNMaxVoxels(Scene *scn, int n)
{
  int last=n,pos;
  Curve *hist = Histogram3(scn);
  pos= hist->n-1;
  while (last>0) {
    if (hist->Y[pos]<0) printf("error");
    if (hist->Y[pos]==0) pos--;
    else {
      hist->Y[pos]--;
      last--;
    }
    if (pos<0) return 0; // just in case
  }
  DestroyCurve(&hist);
  return pos;
}



void ComputeM2(Measures *M, Scene *scn)
{
  Scene *grad = M2_TextGradient3(scn);
  M2_ZeroBorders(grad,2);
  int num = 0.05 * SceneLen(grad);
  Scene *bin = M2_BinarizeByThreshold(grad,M2_ThresholdToGetNMaxVoxels(grad,num));
  DestroyScene(&grad);
  Scene *left=GetLeftAxialHalf(bin);
  Scene *tmp=GetRightAxialHalf(bin);
  Scene *right=FlipSceneAxial(tmp);
  DestroyScene(&tmp);

  int maxx=left->xsize,maxy=left->ysize,maxz=left->zsize;
  if (right->xsize<left->xsize) maxx=right->xsize;
  if (right->ysize<left->ysize) maxy=right->ysize;
  if (right->zsize<left->zsize) maxz=right->zsize;
  Voxel v;
  int pL,pR;
  int MM=0,N=num/2;
    for (v.z=0; v.z < maxz; v.z++)
    for (v.y=0; v.y < maxy; v.y++)
      for (v.x=0; v.x < maxx; v.x++) {
        pL = left->tbz[v.z] + left->tby[v.y] + v.x;
        pR = right->tbz[v.z] + right->tby[v.y] + v.x;
	if (left->data[pL]!=0 && right->data[pR]!=0) MM++;
      }
  float m2= (float)MM/(float)N;
  AddMeasure(M,"M2_GLOBAL",m2);

  int dx[3],dy[5],dz[5];
  dx[0]=0; dx[1]=left->xsize/2; dx[2]=left->xsize-1;
  dy[0]=0; dy[1]=left->ysize/4; dy[2]=left->ysize/2; dy[3]=(3*left->ysize)/4 ; dy[4]=left->ysize;
  dz[0]=0; dz[1]=left->zsize/4; dz[2]=left->zsize/2; dz[3]=(3*left->zsize)/4 ; dz[4]=left->zsize;

  int x,y,z;
  Scene *left_p,*right_p;
  char str[30];
  int counter=1;
  for (x=0;x<=1;x++)
    for (y=0;y<=3;y++)
      for (z=0;z<=3;z++) {
	left_p = ROI3(left,dx[x],dy[y],dz[z],dx[x+1]-1,dy[y+1]-1,dz[z+1]-1);
	right_p = ROI3(right,dx[x],dy[y],dz[z],dx[x+1]-1,dy[y+1]-1,dz[z+1]-1);

	maxx=left_p->xsize,maxy=left_p->ysize,maxz=left_p->zsize;
	if (right_p->xsize<left_p->xsize) maxx=right_p->xsize;
	if (right_p->ysize<left_p->ysize) maxy=right_p->ysize;
	if (right_p->zsize<left_p->zsize) maxz=right_p->zsize;
	Voxel v;
	int pL,pR;
	int MM=0,N=num;
	for (v.z=0; v.z < maxz; v.z++)
	  for (v.y=0; v.y < maxy; v.y++)
	    for (v.x=0; v.x < maxx; v.x++) {
	      pL = left_p->tbz[v.z] + left_p->tby[v.y] + v.x;
	      pR = right_p->tbz[v.z] + right_p->tby[v.y] + v.x;
	      if (left_p->data[pL]!=0 && right_p->data[pR]!=0) MM++;
	    }
	float m2= (float)MM/(float)N;
	sprintf(str,"M2_%03d",counter++);
	AddMeasure(M,str,m2);
	DestroyScene(&left_p);
	DestroyScene(&right_p);
      }

  DestroyScene(&left);
  DestroyScene(&right);

}


